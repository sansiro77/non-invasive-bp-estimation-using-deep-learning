# from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Conv1D, ReLU

# from tensorflow.keras import Model

# def define_LSTM(data_in_shape):
#     X_input = Input(shape=data_in_shape)
#     X = Conv1D(filters=64, kernel_size=5, strides=1, padding='causal', activation='relu')(X_input)
#     X = Bidirectional(LSTM(128, return_sequences=True))(X)
#     X = Bidirectional(LSTM(128, return_sequences=True))(X)
#     X = Bidirectional(LSTM(64, return_sequences=False))(X)
#     X = Dense(512, activation='relu')(X)
#     X = Dense(256, activation='relu')(X)
#     X = Dense(128, activation='relu')(X)

#     X_SBP = Dense(1, name='SBP')(X)
#     X_DBP = Dense(1, name='DBP')(X)

#     model = Model(inputs=X_input, outputs=[X_SBP, X_DBP], name='LSTM')

#     return model

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))

class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.conv1 = CausalConv1d(1, 64, 5)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(64*2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x)).transpose(1,2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:,-1,:].view(-1,64*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x