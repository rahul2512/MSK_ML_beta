import torch as tr 
from torch import nn, sigmoid, tanh,relu

import numpy as np
from torch.nn import Linear 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tensorflow import keras
import sys
from keras.regularizers import l2
from keras.layers import Dense, SimpleRNN, LSTM, GRU
# from ann_visualizer.visualize import ann_viz;

def initiate_RNN_model(inp_dim, out_dim, units, loss, opt, act,final_act, metric, variant):
    model = keras.Sequential()
    if variant == 'SimpleRNN':
        model.add(SimpleRNN(units, input_shape=(None,inp_dim), activation=act)) 
    elif variant == 'LSTM':
        model.add(LSTM(units, input_shape=(None,inp_dim), activation=act)) 
    elif variant == 'GRU':
        model.add(GRU(units, input_shape=(None,inp_dim), activation=act)) 

    model.add(Dense(out_dim, activation=final_act))
    model.compile(loss=loss, optimizer=opt, metrics=metric)
    print("Initialised RNN network")
    return model

loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt="sgd"
metric=["accuracy"]

batch_size = 64
units   = 64
out_dim = 10
inp_dim = 28

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample, sample_label = x_train[0], y_train[0]
act = 'linear'
final_act='linear'
model = initiate_RNN_model(inp_dim, out_dim, units, loss, opt, act, final_act, metric,'SimpleRNN')
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=10)




#### Code below generates a .txt file with row as the list of hypermeters 
#### for a given NN and then that NN hypermeters were cross-validated on cluster
def hyper_param():
# batch-size {Automatic, 64, 1000, 2000}
# epoch = 40
# Adam, SGD, RMSprop
# mse
# relu, tanh, sigmoid
# random_uniform, random_normal, he_normal, xavier, glorot_uniform, glorot_normal (Xavier), 
    with open('hyperparam.txt', 'w') as f:
        print('optim', 'kinit', 'batch_size', 'epoch', 'act', 'num_nodes', 'H_layer', 'metric', 'loss', 'lr', 'p','regularizer_val', file=f)
#        print(optim, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p,reg, file=f)
    return None


def hyper_param_linear():
    return None

