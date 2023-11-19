import numpy as np

import pandas as pd
from pandas import read_csv

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data import random_split

from torch.nn import Linear
from torch.nn import ReLU, LeakyReLU, Softplus
from torch.nn import Module, Sequential
from torch.nn import MaxPool1d, Conv1d, Flatten, BatchNorm1d, Dropout

import torch
from torch import optim

import optuna
import json
import time
import copy
import math

# ==============================================================
# Build CNN Model
# ==============================================================
def CNN1D(optuna_trial, in_channels, out_channels, n_layers, kernel_size, act_func, dropout, n_features, units_linear_output, n_outputs):
    """
    Generate sequential network model with optuna optimization.

    :param optuna_trial: optuna trial class
    :param in_features: num of input nodes
    :param n_layers: num of hidden layers
    :param dropout: perc of final layer dropout
    :param n_output: num of output nodes
    :return: sequential multi layer perceptron model 
    """

    layers = []
    # in_features = Sequential(*layers)(torch.zeros(size=(1, 1, n_features))).shape[1]
    # out_features = int(in_features * units_linear_output)

    for i in range(n_layers): 
        #out_features = optuna_trial.suggest_int("n_units_l{}".format(i), 2, in_features)
        layers.append(Conv1d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size))
        layers.append(act_func)
        layers.append(BatchNorm1d(out_channels))
        layers.append(Dropout(dropout))
        in_channels = out_channels

    layers.append(MaxPool1d(kernel_size=kernel_size)) 
    layers.append(Flatten())
    in_features = Sequential(*layers)(torch.zeros(size=(1, 1, n_features))).shape[1]
    out_features = int(in_features * units_linear_output)

    layers.append(Linear(in_features=in_features, out_features=out_features))
    layers.append(act_func)
    layers.append(BatchNorm1d(num_features=out_features))
    layers.append(Dropout(dropout))
    layers.append(Linear(in_features=out_features, out_features=n_outputs))

    return torch.nn.Sequential(*layers)


# ==============================================================
# The trainning loop
# ==============================================================
def train_model(num_epochs, X, y, params, optuna_trial):

    # start time measurement for training
    # start = time.time()

    # init params before training
    num_features = np.size(X, 1)
    batch_size = 100

    # initialize the MLP model
    cnn_model = CNN1D(optuna_trial,
            in_channels = 1,
            out_channels = params['initial_out_channels'],
            n_layers  = params['n_layers'],
            kernel_size = params['kernel_size'],
            act_func = params['act_func'],
            dropout   = params['dropout'],
            n_features = num_features,
            units_linear_output = params['units_linear_output'],
            n_outputs = 1)
    
    # repare and split datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    # normalize dataset using StandardScaler
    standard_scaler = StandardScaler()
    standard_scaler.fit(X_train)
    X_train = standard_scaler.transform(X_train)
    X_test  = standard_scaler.transform(X_test)

    # transform dataset to tensor
    tensor_X_train = torch.from_numpy(X_train)
    tensor_X_test  = torch.from_numpy(X_test)
    tensor_X_train = tensor_X_train.unsqueeze(1)
    tensor_X_test = tensor_X_test.unsqueeze(1)

    tensor_y_train = torch.from_numpy(y_train).view(len(y_train),1)
    tensor_y_test  = torch.from_numpy(y_test).view(len(y_test),1)

    train_loader = DataLoader(dataset = list(zip(tensor_X_train, tensor_y_train)), batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(dataset = list(zip(tensor_X_test, tensor_y_test)), batch_size=int(batch_size/2), shuffle=True, num_workers=0)

    loss_function = torch.nn.MSELoss()   
    optimizer = getattr(optim, params['optimizer'])(cnn_model.parameters(), lr= params['learning_rate'], weight_decay=params['weight_decay'])

    for epoch in range(num_epochs):

        # for collecting the test loss
        total_test_loss = []
        
        # iterate through training data loader
        for i, (inputs, targets) in enumerate(train_loader):

            # cast the inputs and targets into float
            inputs, targets = inputs.float(), targets.float()

            # make sure the targets reshaped
            targets = targets.reshape((targets.shape[0], 1))
            
            # zero the gradients
            optimizer.zero_grad()

            # perform model forwarding
            outputs = cnn_model(inputs)

            # compute the loss
            loss = loss_function(outputs, targets)

            # perform backward to update the weights
            loss.backward()
    
            # perform optimization
            optimizer.step()  

        # print epoch and loss
        # print ('Epoch {}/{}, loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        # print('----------------------------------------------------\n')

        # model evaluation
        cnn_model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_loader):

                # cast the inputs and targets into float
                inputs, targets = inputs.float(), targets.float()

                # make sure the targets reshaped
                targets = targets.reshape((targets.shape[0], 1))
                
                # perform forward pass
                test_outputs = cnn_model(inputs)
                test_loss = loss_function(test_outputs, targets)
                total_test_loss.append(test_loss.item())
                
    # time_delta = time.time() - start
    # print('Training time in {:.0f}m {:.0f}s'.format(time_delta // 60, time_delta % 60))
    
    return total_test_loss

# ==========================================================
# Objective function for tuning hyper-parameters
# ==========================================================
def objective(X, y, optuna_trial):
    """
    Objective function to run bayesian hyperparameter tuning

    :param trial: optuna study
    :param checkpoint_dir: checkpoint dir args
    :param cfg: config file
    :return: mean RMSE test loss
    """ 

    # for tuning samples 
    params = {
              'learning_rate': optuna_trial.suggest_float('learning_rate', 1e-6, 1e-2), 
              'optimizer': optuna_trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
              'weight_decay': optuna_trial.suggest_float('weight_decay', 1e-4, 1e-2),
              'initial_out_channels': optuna_trial.suggest_int('initial_out_channels', 4, 8),
              "n_layers" : optuna_trial.suggest_int("n_layers", 1, 5),
              'kernel_size': optuna_trial.suggest_int('kernel_size', 2, 8),
              'act_func': optuna_trial.suggest_categorical('act_func', ['ReLU', 'LeakyReLU']),
              "dropout" : optuna_trial.suggest_float('dropout', 0.1, 0.5),
              'units_linear_output': optuna_trial.suggest_float('optuna_trial.suggest_float', 0.2, 0.5)
              }
    
    # num epochs for training
    num_epochs = 2

    # call the train model
    test_loss = train_model(num_epochs, X, y, params, optuna_trial) 

    # return the mean MSE loss
    mean_loss = np.mean(test_loss)

    # sumarize loss values
    print("Summary: max_loss={}, min_loss={}, avg_loss={} \n".format(np.max(test_loss), np.min(test_loss), mean_loss))

    return mean_loss

# ==============================================================
# Call and train model
# ==============================================================
def train_and_tune_CNN(X, y):

    # init optuna tuning object
    num_trials = 10
    search_space = optuna.create_study(direction ="minimize", sampler=optuna.samplers.TPESampler())
    search_space.optimize(lambda trial: objective(X, y, trial), n_trials=num_trials)

    model_params = search_space.best_trial.params
    model_params['optuna_best_trial_number'] =  search_space.best_trial.number 
    model_params['optuna_best_trial_value'] = float(np.round(search_space.best_value, 6))
    model_params["n_trials"] = num_trials

    with open(f"/Users/nghihuynh/Documents/MscTUM_BioTech/4th_semester/Internship/intern_pheno_nnets/tuning/tuning_cnn_" + str(num_trials) + ".json", 'w') as fp:
        json.dump(model_params, fp)
    
    print()
    print('----------------------------------------------------')
    print("Tuning MLP model with Optuna: ")
    print("The result is writen at ./tuning/" + "tuning_cnn_model_" + str(num_trials) + ".json")
    print('----------------------------------------------------\n')

    return 0
