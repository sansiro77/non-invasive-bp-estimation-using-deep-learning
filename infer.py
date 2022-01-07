#

from os.path import join
from functools import partial

import tensorflow as tf

def read_tfrecord(example, win_len=875):
    tfrecord_format = (
        {
            'ppg': tf.io.FixedLenFeature([win_len], tf.float32),
            'label': tf.io.FixedLenFeature([2], tf.float32)
        }
    )
    parsed_features = tf.io.parse_single_example(example, tfrecord_format)

    return parsed_features['ppg'], (parsed_features['label'][0], parsed_features['label'][1])

def create_dataset(tfrecords_dir, tfrecord_basename, win_len=875, batch_size=32, modus='train'):

    pattern = join(tfrecords_dir, modus, tfrecord_basename + "_" + modus + "_?????_of_?????.tfrecord")
    dataset = tf.data.TFRecordDataset.list_files(pattern)
    

    if modus == 'train':
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=800,
            block_length=400)
    else:
        dataset = dataset.interleave(
            tf.data.TFRecordDataset)

    dataset = dataset.map(partial(read_tfrecord, win_len=875), num_parallel_calls=2)
    dataset = dataset.shuffle(4096, reshuffle_each_iteration=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.repeat()

    return dataset


import tensorflow.keras as ks
from kapre import STFT, Magnitude, MagnitudeToDecibel

dependencies = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel}

model = ks.models.load_model('ckpts/lstm_ppg_nonmixed.h5', custom_objects=dependencies)
test_dataset = create_dataset(tfrecords_dir='/data/ppg_dataset', tfrecord_basename='MIMIC_III_ppg', win_len=875, batch_size=32, modus='test')

test_dataset = iter(test_dataset)
for i in range(int(2.5e5//32)):

    ppg_test, BP_true = test_dataset.next()
    print('ppg_test:\n', ppg_test)
    BP_est = model.predict(ppg_test)
    print('SBP_est:\n', BP_est[0])
    print('SBP_true:\n', BP_true[0])
    print('DBP_est:\n', BP_est[1])
    print('DBP_true:\n', BP_true[1])
    break






