""" train neural architectures using PPG data

This script trains a neural network using PPG data. The data is loaded from the the .tfrecord files created by the script
'hdf_to_tfrecord.py'. Four different neural architectures can be selected:

- AlexNet [1]
- ResNet [2]
- Architecture published by Slapnicar et al. (modified to work with Tensorflow 2.4.1) [3] The original code can be downloaded
  from https://github.com/gslapnicar/bp-estimation-mimic3
- LSTM network

A checkpoint callback is used to store the best network weights in terms of validation loss. These weights are subsequently
used to perform predictions on the test set. Test results are stored in a csv file for later evaluation.

References
[1] A. Krizhevsky, I. Sutskever, und G. E. Hinton, „ImageNet classification with deep convolutional neural networks“,
    Commun. ACM, Bd. 60, Nr. 6, S. 84–90, Mai 2017, doi: 10.1145/3065386.
[2] K. He, X. Zhang, S. Ren, und J. Sun, „Deep Residual Learning for Image Recognition“, in 2016 IEEE Conference on
    Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, Juni 2016, S. 770–778. doi: 10.1109/CVPR.2016.90.
[3] G. Slapničar, N. Mlakar, und M. Luštrek, „Blood Pressure Estimation from Photoplethysmogram Using a Spectro-Temporal
    Deep Neural Network“, Sensors, Bd. 19, Nr. 15, S. 3420, Aug. 2019, doi: 10.3390/s19153420.

File: prepare_MIMIC_dataset.py
Author: Dr.-Ing. Fabian Schrumpf
E-Mail: Fabian.Schrumpf@htwk-leipzig.de
Date created: 8/9/2021
Date last modified: 8/9/2021
"""

from os.path import expanduser, isdir, join
from os import environ, mkdir
from sys import argv
from functools import partial
from datetime import datetime
import argparse

import tensorflow as tf
import pandas as pd
import numpy as np

# gpu_devices = tf.config.experimental.list_physical_devices("GPU")
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

from models.define_ResNet_1D import ResNet50_1D
from models.define_AlexNet_1D import AlexNet_1D
from models.define_LSTM import LSTM

# from models.slapnicar_model import raw_signals_deep_ResNet

import matplotlib.pyplot as plt
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MyDataSet(Dataset):
    def __init__(self, X, Y, ID):
        self.X = X
        self.Y = Y
        self.ID = ID

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.ID[index]

def create_dataset(h5_dir, basename, win_len=875, batch_size=32, N_train=512, N_val=128, N_test=128,
                    divide_by_subject=True):

    N_train = int(N_train)
    N_val = int(N_val)
    N_test = int(N_test)

    h5_file = h5_dir + basename + '.h5'
    csv_path = h5_dir + 'csv_record/'
    if not isdir(csv_path):
        mkdir(csv_path)

    with h5py.File(h5_file, 'r') as f:
        BP = np.array(f.get('/label'))
        BP = np.round(BP)
        BP = np.transpose(BP) # comment for rppg
        subject_idx = np.squeeze(np.array(f.get('/subject_idx')))

    N_samp_total = BP.shape[1] # ppg
    # N_samp_total = BP.shape[0] # rppg
    subject_idx = subject_idx[:N_samp_total]

    # Divide the dataset into training, validation and test set
    # -------------------------------------------------------------------------------
    if divide_by_subject is True:
        valid_idx = np.arange(subject_idx.shape[-1])

        # divide the subjects into training, validation and test subjects
        subject_labels = np.unique(subject_idx)
        subjects_train_labels, subjects_val_labels = train_test_split(subject_labels, test_size=0.5)
        subjects_val_labels, subjects_test_labels = train_test_split(subjects_val_labels, test_size=0.5)

        # Calculate samples belong to training, validation and test subjects
        train_part = valid_idx[np.isin(subject_idx,subjects_train_labels)]
        val_part = valid_idx[np.isin(subject_idx,subjects_val_labels)]
        test_part = valid_idx[np.isin(subject_idx, subjects_test_labels)]

        # draw a number samples defined by N_train, N_val and N_test from the training, validation and test subjects
        idx_train = np.random.choice(train_part, N_train, replace=False)
        idx_val = np.random.choice(val_part, N_val, replace=False)
        idx_test = np.random.choice(test_part, N_test, replace=False)
    else:
        # Create a subset of the whole dataset by drawing a number of subjects from the dataset. The total number of
        # samples contributed by those subjects must equal N_train + N_val + _N_test
        subject_labels, SampSubject_hist = np.unique(subject_idx, return_counts=True)
        cumsum_samp = np.cumsum(SampSubject_hist)
        subject_labels_train = subject_labels[:np.nonzero(cumsum_samp>(N_train+N_val+N_test))[0][0]]
        idx_valid = np.nonzero(np.isin(subject_idx,subject_labels_train))[0]

        # divide subset randomly into training, validation and test set
        idx_train, idx_val = train_test_split(idx_valid, train_size= N_train, test_size=N_val+N_test)
        idx_val, idx_test = train_test_split(idx_val, test_size=0.5)

    # save ground truth BP values of training, validation and test set in csv-files for future reference
    BP_train = BP[:,idx_train]
    d = {"SBP": np.transpose(BP_train[0, :]), "DBP": np.transpose(BP_train[1, :])}
    train_set = pd.DataFrame(d)
    train_set.to_csv(csv_path + basename + '_trainset.csv')
    BP_val = BP[:,idx_val]
    d = {"SBP": np.transpose(BP_val[0, :]), "DBP": np.transpose(BP_val[1, :])}
    train_set = pd.DataFrame(d)
    train_set.to_csv(csv_path + basename + '_valset.csv')
    BP_test = BP[:,idx_test]
    d = {"SBP": np.transpose(BP_test[0, :]), "DBP": np.transpose(BP_test[1, :])}
    train_set = pd.DataFrame(d)
    train_set.to_csv(csv_path + basename + '_testset.csv')

    with h5py.File(h5_file, 'r') as f:
        # load ppg and BP data as well as the subject numbers the samples belong to
        ppg_h5 = np.array(f.get('/ppg'))
        # ppg_h5 = np.array(f.get('/rppg'))
        BP = np.array(f.get('/label'))
        subject_idx = np.array(f.get('/subject_idx'))

        # rppg
        # ppg_h5 = np.transpose(ppg_h5)
        # BP = np.transpose(BP)
        # subject_idx = np.transpose(subject_idx)

        ppg_h5 = torch.from_numpy(ppg_h5).view(-1,1,win_len).float()
        BP = torch.from_numpy(BP).view(-1,2).float()
        subject_idx = torch.from_numpy(subject_idx).view(-1,1)
        print("Data Set")
        print(ppg_h5.shape)
        print(BP.shape)

        dataset_train = MyDataSet(ppg_h5[idx_train,:], BP[idx_train,:], subject_idx[idx_train,:])
        dataset_val = MyDataSet(ppg_h5[idx_val,:], BP[idx_val,:], subject_idx[idx_val,:])
        dataset_test = MyDataSet(ppg_h5[idx_test,:], BP[idx_test,:], subject_idx[idx_test,:])

    return dataset_train, dataset_val, dataset_test

def get_model(architecture, input_shape, UseDerivative=False):
    print('debug architecture', architecture)
    return {
        'resnet': ResNet50_1D(input_shape, UseDerivative=UseDerivative),
        'alexnet': AlexNet_1D(input_shape, UseDerivative=UseDerivative),
        # 'slapnicar' : raw_signals_deep_ResNet(input_shape, UseDerivative=UseDerivative),
        'lstm' : LSTM()
    }[architecture]

def ppg_train_mimic_iii(architecture,
                        data_dir,
                        results_dir,
                        CheckpointDir,
                        tensorboard_tag,
                        basename,
                        experiment_name,
                        win_len=875,
                        batch_size=32,
                        lr = None,
                        N_epochs = 20,
                        Ntrain=512,
                        Nval=128,
                        Ntest=128,
                        UseDerivative=False,
                        earlystopping=True):
    
    PathSaveLeastLoss = CheckpointDir+basename+"_leastLoss.pth"
    PathSaveBestValid = CheckpointDir+basename+"_bestValid.pth"

    RunningLossSave = np.array([])
    ValidationLossSave = np.array([])

    if not isdir(results_dir):
        mkdir(results_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_dataset, val_dataset, test_dataset = create_dataset(data_dir, basename, 
                                                                win_len=win_len, batch_size=batch_size)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    data_in_shape = (win_len,1)

    # load the neurarchitecture
    model = get_model(architecture, data_in_shape, UseDerivative=UseDerivative).to(device)
    # print(model.summary())

    # callback for logging training and validation results
    csvLogger_cb = tf.keras.callbacks.CSVLogger(
        filename=join(results_dir,experiment_name + '_learningcurve.csv')
    )

    # checkpoint callback
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=join(CheckpointDir , experiment_name + '_cb.h5'),
        save_best_only=True
    )

    # tensorboard callback
    tensorbard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=join(results_dir, 'tb', tensorboard_tag),
        histogram_freq=0,
        write_graph=False
    )

    # callback for early stopping if validation loss stops improving
    EarlyStopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # define Adam optimizer
    if lr is None:
        optimizer = optim.Adam(model.parameters())
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # compile model using mean squared error as loss function
    criterion = nn.MSELoss()
    # model.compile(
    #     optimizer=opt,
    #     loss = tf.keras.losses.mean_squared_error,
    #     metrics = [['mae'], ['mae']]
    # )

    cb_list = [checkpoint_cb,
               tensorbard_cb,
               csvLogger_cb,
               EarlyStopping_cb if earlystopping == True else []]

    # Perform Training and Validation
    # history = model.fit(
    #     train_dataset,
    #     steps_per_epoch=Ntrain//batch_size,
    #     epochs=N_epochs,
    #     validation_data=val_dataset,
    #     validation_steps=Nval//batch_size,
    #     callbacks=cb_list
    # )
    num_batches = int(Ntrain/batch_size)
    for epoch in range(N_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, indices = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i % num_batches == (num_batches - 1)) and (epoch%1 == 0):
                print("[%d, %5d] loss: %.4f" %
                      (epoch + 1, i + 1, running_loss/num_batches))
                RunningLossSave = np.append(RunningLossSave, running_loss/num_batches)
                if epoch == 0:
                    LeastLoss = running_loss/num_batches
                    torch.save(model.state_dict(), PathSaveLeastLoss)
                if running_loss/num_batches < LeastLoss:
                    LeastLoss = running_loss/num_batches
                    torch.save(model.state_dict(), PathSaveLeastLoss)
                running_loss = 0.0

        if epoch%1 == 0:
            model.eval()
            total = 0
            running_loss = 0.0
            with torch.no_grad():
                for num, data in enumerate(valloader, 0):
                    inputs, labels, indices = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    
                    loss = criterion(outputs, labels)
                    total += labels.size(0)
                    running_loss += loss.item()

            ValidationLoss = running_loss/(total/batch_size)
            ValidationLossSave = np.append(ValidationLossSave, ValidationLoss)
            if epoch == 0:
                    LeastValidLoss = ValidationLoss
                    torch.save(model.state_dict(), PathSaveBestValid)
            if ValidationLoss < LeastValidLoss:
                print("Accuracy on the test set %d: loss: %.4f" % (total, ValidationLoss))
                if ValidationLoss < LeastValidLoss:
                    LeastValidLoss = ValidationLoss
                    torch.save(model.state_dict(), PathSaveBestValid)

    # Predictions on the testset
    # model.load_weights(checkpoint_cb.filepath)
    model.load_state_dict(torch.load(PathSaveBestValid))
    test_results = pd.DataFrame({'SBP_true' : [],
                                 'DBP_true' : [],
                                 'SBP_est' : [],
                                 'DBP_est' : []})

    # store predictions on the test set as well as the corresponding ground truth in a csv file
    # test_dataset = iter(test_dataset)
    # for i in range(int(Ntest//batch_size)):
    #     ppg_test, BP_true = test_dataset.next()
    #     BP_est = model.predict(ppg_test)
    #     print('debug BP_est', BP_est)
    #     TestBatchResult = pd.DataFrame({'SBP_true' : BP_true[0].numpy(),
    #                                     'DBP_true' : BP_true[1].numpy(),
    #                                     'SBP_est' : np.squeeze(BP_est[0]),
    #                                     'DBP_est' : np.squeeze(BP_est[1])})
    #     test_results = test_results.append(TestBatchResult)

    with torch.no_grad():
        for num, data in enumerate(testloader, 0):
            inputs, labels, indices = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            print("Test Data")
            print(labels.shape)
            print(outputs.shape)
            TestBatchResult = pd.DataFrame({'SBP_true' : labels[:,0].numpy(),
                                        'DBP_true' : labels[:,1].numpy(),
                                        'SBP_est' : np.squeeze(outputs[:,0]),
                                        'DBP_est' : np.squeeze(outputs[:,1])})
            test_results = test_results.append(TestBatchResult)

    ResultsFile = join(results_dir,experiment_name + '_test_results.csv')
    test_results.to_csv(ResultsFile)

    # idx_min = np.argmin(history.history['val_loss'])

    print(' Training finished')
    return
    # return history.history['SBP_mae'][idx_min], history.history['DBP_mae'][idx_min], history.history['val_SBP_mae'][idx_min], history.history['val_DBP_mae'][idx_min]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str, help="unique name for the training")
    parser.add_argument('data_dir', type=str, help="directory containing the train, val and test subdirectories containing tfrecord files")
    parser.add_argument('results_dir', type=str, help="directory in which results are stored")
    parser.add_argument('ckpts_dir', type=str, help="directory used for storing model checkpoints")
    parser.add_argument('--arch', type=str, default="alexnet", help="neural architecture used for training (alexnet (default), resnet,  slapnicar, lstm)")
    parser.add_argument('--lr', type=float, default=0.003, help="initial learning rate (default: 0.003)")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size used for training (default: 32)")
    parser.add_argument('--win_len', type=int, default=875, help="length of the ppg windows in samples (default: 875)")
    parser.add_argument('--epochs', type=int, default=60, help="maximum number of epochs for training (default: 60)")
    parser.add_argument('--gpuid', type=str, default=None, help="GPU-ID used for training in a multi-GPU environment (default: None)")
    args = parser.parse_args()

    if len(argv) > 1:
        architecture = args.arch
        experiment_name = args.experiment_name
        experiment_name = datetime.now().strftime("%Y-%d-%m") + '_' + architecture + '_' + experiment_name
        data_dir = args.data_dir
        results_dir = args.results_dir
        CheckpointDir = args.ckpts_dir
        tb_tag = experiment_name
        lr = args.lr if args.lr > 0 else None
        batch_size = args.batch_size
        win_len = args.win_len
        N_epochs = args.epochs
        if args.gpuid is not None:
            environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    else:
        architecture = 'lstm'
        experiment_name = datetime.now().strftime("%Y-%d-%m") + '_' + architecture + '_' + 'mimic_iii_ppg_nonmixed_pretrain'
        HomePath = expanduser("~")
        data_dir = join(HomePath,'data','MIMIC-III_BP', 'tfrecords_nonmixed')
        results_dir = join(HomePath,'data','MIMIC-III_BP', 'results')
        CheckpointDir = join(HomePath,'data','MIMIC-III_BP', 'checkpoints')
        tb_tag = architecture + '_' + 'mimic_iii_ppg_pretrain'
        batch_size = 64
        win_len = 875
        lr = None
        N_epochs = 60

    basename = 'MIMIC_III_ppg'
    # basename = 'rPPG-BP-UKL_rppg_7s'

    ppg_train_mimic_iii(architecture,
                        data_dir,
                        results_dir,
                        CheckpointDir,
                        tb_tag,
                        basename,
                        experiment_name,
                        win_len=win_len,
                        batch_size=batch_size,
                        lr=lr,
                        N_epochs=N_epochs,
                        UseDerivative=True,
                        earlystopping=False)