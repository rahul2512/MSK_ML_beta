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
# from ann_visualizer.visualize import ann_viz;


### Initite NN model
def initiate_NN_model(inp_dim,out_dim,nbr_Hlayer,Neu_layer,activation,p_drop,lr,optim,loss,metric,kinit,final_act,regularizer_val):
    model = keras.Sequential()
    model.add(keras.layers.Dense(Neu_layer, input_shape=(inp_dim,), activation=activation))
    for i in range(nbr_Hlayer):
        model.add(keras.layers.Dense(Neu_layer, activation=activation,kernel_initializer=kinit))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(p_drop))
    model.add(keras.layers.Dense(out_dim, activation=final_act))
    try:
        opt = optim(learning_rate=lr)
    except:
        opt = optim(lr=lr)
    model.compile(loss=loss, optimizer=opt, metrics=metric)
    print("Initialised NN network")
    return model

def initiate_Linear_model(inp_dim,out_dim,nbr_Hlayer,Neu_layer,activation,p_drop,lr,optim,loss,metric,kinit,final_act,regularizer_val):
    #### rest of the parameters are redundnt but kept for generalisibilty of code
    # model = keras.Sequential()
    # model.add(keras.layers.Dense(out_dim, input_shape=(inp_dim,), activation='linear'))
    try:
        opt = optim(learning_rate=lr)
    except:
        opt = optim(lr=lr)
    inputs = keras.layers.Input(shape=(inp_dim,))
    outputs = keras.layers.Dense(out_dim)(inputs)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer=opt, metrics=metric)
    print("Initialised Linear network")
    return model


def initiate_LR_model(inp_dim,out_dim,nbr_Hlayer,Neu_layer,activation,p_drop,lr,optim,loss,metric,kinit,final_act,regularizer_val):
    #### rest of the parameters are redundnt but kept for generalisibilty of code
    model = keras.Sequential()
    model.add(keras.layers.Dense(out_dim, input_shape=(inp_dim,), activation='sigmoid'))
    try:
        opt = optim(learning_rate=lr)
    except:
        opt = optim(lr=lr)
    model.compile(loss=loss, optimizer=opt, metrics=metric)
    print("Initialised logistic regression network")
    return model



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
        for optim in ['Adam', 'RMSprop', 'SGD']:
            for kinit in ['glorot_normal','random_normal', 'he_normal']:
                for batch_size in [64,256,1028]:
                    for epoch in [50,100,200]:
                        for act in ['relu','tanh','sigmoid']:
                            for H_layer in [1,2,4,6,8,10]:
                                for metric in ['mse']:
                                    for loss in ['mse']:
                                        for lr in [0.001,0.005]:
                                            for p in [0,0.2]:
                                                for num_nodes in np.arange(200,2100,200):
                                                    for reg in [0]:
                                                        print(optim, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p,reg, file=f)
    return None


def hyper_param_linear():
# batch-size {Automatic, 64, 1000, 2000}
# epoch = 40
# Adam, SGD, RMSprop
# mse
# relu, tanh, sigmoid
# random_uniform, random_normal, he_normal, xavier, glorot_uniform, glorot_normal (Xavier), 
    with open('hyperparam_linear.txt', 'w') as f:
        print('optim', 'kinit', 'batch_size', 'epoch', 'act', 'num_nodes', 'H_layer', 'metric', 'loss', 'lr', 'p','regularizer_val', file=f)
        for optim in ['Adam', 'RMSprop', 'SGD']:
            for kinit in ['glorot_normal','random_normal', 'he_normal']:
                for batch_size in [64,256]:
                    for epoch in [50,100,200]:
                        for act in ['linear']:
                            for H_layer in [0]:
                                for metric in ['mse']:
                                    for loss in ['mse']:
                                        for lr in [0.001]:
                                            for p in [0]:
                                                for num_nodes in [200]:
                                                    for reg in [0]:
                                                        print(optim, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p,reg, file=f)
    return None

