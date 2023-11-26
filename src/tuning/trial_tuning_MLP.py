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

import pickle as pk

from torch.nn import MSELoss
from torch.nn import Sequential, MaxPool1d, Flatten, LeakyReLU, BatchNorm1d, Dropout, Linear, ReLU, Sigmoid
from torch.optim import SGD, Adam

import torch
from torch import optim

import optuna
import json
import time
import copy
import math


# ==============================================================
# Build MLP Model
# ==============================================================
def MLP(optuna_trial, in_features, n_layers, dropout, activation, n_outputs):
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
    fc_layer = in_features

    for i in range(n_layers): 
        # print('In_features', in_features)
        out_features = optuna_trial.suggest_int("n_units_l{}".format(i), 33, 231)
        # print('Out_features', out_features)
        layers.append(Linear(in_features, out_features))
        if activation == 'relu':
            layers.append(ReLU())
        else:
            layers.append(LeakyReLU())     
        in_features = out_features

    layers.append(Dropout(dropout))
    layers.append(Linear(in_features, n_outputs))

    return Sequential(*layers)
    
# ==============================================================
# The trainning loop
# ==============================================================
def train_model(num_epochs, X, y, k_folds, batch_size, params, optuna_trial):
    
    # number of folds for cross-validation
    kfold = KFold(n_splits=k_folds)

    # declare arrays for storing total loss and other metrics
    train_loss_total = []
    val_loss_total = []

    # for collecting the test loss
    total_test_loss = []


    for fold, (train_ids, val_ids) in enumerate(kfold.split(X, y)):
        # print('FOLD {}: len(train)={}, len(val)={}'.format(fold, len(train_ids), len(val_ids)))

        # extract X, y for training and validating
        X_train, y_train = X[train_ids], y[train_ids]
        X_val, y_val = X[val_ids], y[val_ids]

        # PCA
        pca = PCA(params['pca'])
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_val = pca.transform(X_val)
        pk.dump(pca, open('./pca.pkl', 'wb'))
        # print('shape after PCA: train ={}, val={}'.format(X_train.shape, X_val.shape))

        # Define relevant hyperparameter for the ML task
        num_features = np.size(X_train, 1) # len of column

        # transform to tensor 
        tensor_X_train, tensor_y_train = torch.Tensor(X_train), torch.Tensor(y_train)
        tensor_X_val, tensor_y_val = torch.Tensor(X_val), torch.Tensor(y_val)
        
        # Define data loaders for training and testing data in this fold
        train_loader = DataLoader(dataset=list(zip(tensor_X_train, tensor_y_train)), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=list(zip(tensor_X_val, tensor_y_val)), batch_size=batch_size, shuffle=True) 

         # Call model
        model = MLP(optuna_trial,
            in_features = num_features,
            n_layers  = params['n_layers'],
            dropout   = params['dropout'],
            activation = params['activation'],
            n_outputs = 1)
        
         # define loss function and optimizer
        criterion = MSELoss()   
        optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'], weight_decay=params['weight_decay'])
        
        for epoch in range(num_epochs):
            
            # for collecting the test loss
            total_test_loss = []
            
            # iterate through training data loader
            for i, (inputs, targets) in enumerate(train_loader):
                # call model train
                    model.train()
                    # get predicted outputs
                    pred_outputs = model(inputs)
                    # calculate loss
                    loss_training = criterion(pred_outputs, targets)
                    # optimizer sets to 0 gradients
                    optimizer.zero_grad()
                    # set the loss to back propagate through the network updating the weights
                    loss_training.backward()
                    # perform optimizer step
                    optimizer.step() 

            # model evaluation
            model.eval()
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(val_loader):

                    # cast the inputs and targets into float
                    inputs, targets = inputs.float(), targets.float()

                    # make sure the targets reshaped
                    targets = targets.reshape((targets.shape[0], 1))
                    
                    # perform forward pass
                    test_outputs = model(inputs)
                    test_loss = criterion(test_outputs, targets)
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
              'optimizer': optuna_trial.suggest_categorical("optimizer", ["Adam", "SGD"]),
              'weight_decay': optuna_trial.suggest_float('weight_decay', 1e-4, 1e-2),
              'activation': optuna_trial.suggest_categorical('activation', ['leakyrelu', 'relu']),
              'n_layers' : optuna_trial.suggest_int("n_layers", 1, 5),
              'dropout' : optuna_trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
              'pca': optuna_trial.suggest_float('pca', 0.7, 0.95, step=0.05)
              }
    
    # num epochs for training
    num_epochs = 2
    batch_size = 50
    k_folds = 5

    # call the train model
    test_loss = train_model(num_epochs, X, y, k_folds, batch_size, params, optuna_trial) 

    # return the mean MSE loss
    mean_loss = np.mean(test_loss)

    # sumarize loss values
    print("Summary: max_loss={}, min_loss={}, avg_loss={} \n".format(np.max(test_loss), np.min(test_loss), mean_loss))

    return mean_loss


# ==============================================================
# Call and train model
# ==============================================================
def trial_train_and_tune_MLP(datapath, X, y):

    # init optuna tuning object
    num_trials = 10
    search_space = optuna.create_study(direction ="minimize", sampler=optuna.samplers.TPESampler())
    search_space.optimize(lambda trial: objective(X, y, trial), n_trials=num_trials)

    model_params = search_space.best_trial.params
    model_params['optuna_best_trial_number'] =  search_space.best_trial.number 
    model_params['optuna_best_trial_value'] = float(np.round(search_space.best_value, 6))
    model_params["n_trials"] = num_trials

    with open(f"./tuning_mlp_model_with_optuna_num_trials_" + str(num_trials) + ".json", 'w') as fp:
        json.dump(model_params, fp)
    
    print()
    print('----------------------------------------------------')
    print("Tuning MLP model with Optuna: ")
    print("The result is writen at ./tuning/" + "tuning_mlp_model_with_optuna_num_trials_" + str(num_trials) + ".json")
    print('----------------------------------------------------\n')

    return 0