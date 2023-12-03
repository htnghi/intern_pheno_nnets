import numpy as np

import pandas as pd
from pandas import read_csv

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data import random_split

import pickle as pk

from torch.nn import MSELoss
from torch.nn import Sequential, MaxPool1d, Flatten, LeakyReLU, BatchNorm1d, Dropout, Linear, ReLU, Tanh
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
def MLP(optuna_trial, in_features, n_layers, initial_outfeatures_factor, dropout, activation, n_outputs):
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
        # out_features = optuna_trial.suggest_int("n_units_l{}".format(i), 30, 300)
        # out_features = optuna_trial.suggest_int("n_units_l{}".format(i), 30, 7000)
        out_features = int(in_features * initial_outfeatures_factor)

        # print('Out_features', out_features)
        layers.append(Linear(in_features, out_features))
        if activation == 'relu':
            layers.append(ReLU())
        elif activation == 'leakyrelu':
            layers.append(LeakyReLU())
        else:
            layers.append(Tanh())     
        in_features = out_features

    layers.append(Dropout(dropout))
    layers.append(Linear(in_features, n_outputs))

    return Sequential(*layers)
    
# ==============================================================
# The trainning loop
# ==============================================================
def train_model(num_epochs, X, y, batch_size, params, optuna_trial):

    # declare arrays for storing total loss and other metrics
    arr_val_losses = []
    arr_r2_scores = []
    arr_exp_vars = []

    # split X, y for training and validating
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, shuffle=True)

    # For onehot encoding
    X_train, X_val = X_train.reshape(X_train.shape[0], -1), X_val.reshape(X_val.shape[0], -1)

    # MinMax Scaler
    minmax_scaler = MinMaxScaler()
    y_train = np.expand_dims(y_train, axis=1)
    y_train_scaled = minmax_scaler.fit_transform(y_train)
    y_val = np.expand_dims(y_val, axis=1)
    y_val_scaled = minmax_scaler.fit_transform(y_val)

    # Normalize dataset using StandardScaler
    # standard_scaler = StandardScaler()
    # standard_scaler.fit(X_train)
    # X_train = standard_scaler.transform(X_train)
    # X_val = standard_scaler.transform(X_val)

    # PCA
    # pca = PCA(params['pca'])
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    # X_val = pca.transform(X_val)
    # pk.dump(pca, open('./pca.pkl', 'wb'))
    # print('shape after PCA: train ={}, val={}'.format(X_train.shape, X_val.shape))
        
    # get the number of features
    num_features = np.size(X_train, 1)

    # transform to tensor 
    tensor_X_train, tensor_y_train = torch.Tensor(X_train), torch.Tensor(y_train_scaled)
    tensor_X_val, tensor_y_val = torch.Tensor(X_val), torch.Tensor(y_val_scaled)
        
    # define data loaders for training and testing data in this fold
    train_loader = DataLoader(dataset=list(zip(tensor_X_train, tensor_y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=list(zip(tensor_X_val, tensor_y_val)), batch_size=batch_size, shuffle=True) 

    # creat the model object
    model = MLP(optuna_trial,
        in_features = num_features,
        n_layers  = params['n_layers'],
        initial_outfeatures_factor = params['initial_outfeatures_factor'],
        dropout   = params['dropout'],
        activation = params['activation'],
        n_outputs = 1)
        
    # define loss function and optimizer
    criterion = MSELoss()   
    optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'], weight_decay=params['weight_decay'])
    
    try:   
        for epoch in range(num_epochs):
            
            # iterate through training data loader
            for i, (inputs, targets) in enumerate(train_loader):
                model.train()
                pred_outputs = model(inputs)
                loss_training = criterion(pred_outputs, targets)
                optimizer.zero_grad()
                loss_training.backward()
                optimizer.step() 

            # model evaluation in each epoch
            epoch_val_losses = []
            epoch_val_expvars = []
            epoch_val_r2scors = []

            model.eval()
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(val_loader):

                    # cast the inputs and targets into float
                    inputs, targets = inputs.float(), targets.float()

                    # make sure the targets reshaped
                    targets = targets.reshape((targets.shape[0], 1))
                    
                    # perform forward pass
                    test_outputs = model(inputs)

                    # record the loss for each step to calculate the avg afterward
                    test_loss = criterion(test_outputs, targets)
                    epoch_val_losses.append(test_loss.item())

                    # calculate the exp_var to check during optuna tuning
                    np_targets = targets.squeeze().numpy()
                    np_predics = test_outputs.detach().squeeze().numpy()
                    
                    test_expvar = explained_variance_score(np_targets, np_predics, force_finite=False)
                    test_r2scor = r2_score(np_targets, np_predics)
                    epoch_val_expvars.append(test_expvar)
                    epoch_val_r2scors.append(test_r2scor)
            
            epoch_avg_loss = np.average(epoch_val_losses)
            arr_val_losses.append(epoch_avg_loss)

            # check the explained variance and r2score of validation phase after each epoch
            epoch_avg_expvar = np.average(epoch_val_expvars)
            epoch_avg_r2scor = np.average(epoch_val_r2scors)
            print("Validation phase, epoch {}: avg_expvar={:.3f}, avg_r2score={:.3f}, avg_mseloss={:.3f}".format(epoch, 
                                                epoch_avg_expvar, epoch_avg_r2scor, epoch_avg_loss))
            
            # try to tune with r2_score
            arr_r2_scores.append(epoch_avg_r2scor)
            arr_exp_vars.append(epoch_avg_expvar)
                    
        # time_delta = time.time() - start
        # print('Training time in {:.0f}m {:.0f}s'.format(time_delta // 60, time_delta % 60))
        print("--------------------------------------------------------------------")
        print("")
        
        return arr_val_losses
        # return arr_r2_scores
        # return arr_exp_vars

    except ValueError as e:
        # Check if the error is related to NaN values
        if "Input contains NaN" in str(e):
            raise optuna.exceptions.TrialPruned()
        else:
            # If it's another type of ValueError, raise it again
            raise

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
    print("")

    # for tuning samples 
    params = {
              'learning_rate': optuna_trial.suggest_float('learning_rate', 1e-6, 1e-2), 
              'optimizer': optuna_trial.suggest_categorical("optimizer", ["Adam", "SGD"]),
              'weight_decay': optuna_trial.suggest_float('weight_decay', 1e-8, 1e-2),
              'initial_outfeatures_factor': optuna_trial.suggest_float('initial_outfeatures_factor', 0.01, 0.8, step=0.01),
              'activation': optuna_trial.suggest_categorical('activation', ['leakyrelu', 'relu', 'tanh']),
              'n_layers' : optuna_trial.suggest_int("n_layers", 1, 5),
              'dropout' : optuna_trial.suggest_float('dropout', 0.1, 0.5, step=0.05),
              'pca': optuna_trial.suggest_float('pca', 0.7, 0.95, step=0.05)
              }
    
    # num epochs for training
    num_epochs = 5
    batch_size = 50

    # call the train model
    val_loss = train_model(num_epochs, X, y, batch_size, params, optuna_trial) 
    # r2_score = train_model(num_epochs, X, y, batch_size, params, optuna_trial)
    # exp_var = train_model(num_epochs, X, y, batch_size, params, optuna_trial)

    # return the mean MSE loss
    mean_loss = np.mean(val_loss)
    # mean_r2score = np.mean(r2_score)
    # mean_expvar = np.mean(exp_var)

    # sumarize loss values
    print("Summary: max_loss={}, min_loss={}, avg_loss={} \n".format(np.max(val_loss), np.min(val_loss), mean_loss))
    # print("Summary: max_r2score={}, min_r2score={}, avg_r2score={} \n".format(np.max(r2_score), np.min(r2_score), mean_r2score))
    # print("Summary: max_expvar={}, min_expvar={}, avg_expvar={} \n".format(np.max(exp_var), np.min(exp_var), mean_expvar))

    return mean_loss
    # return mean_r2score
    # return mean_expvar


# ==============================================================
# Call and train model
# ==============================================================
def trial_train_and_tune_MLP(datapath, X, y):

    # init optuna tuning object
    num_trials = 15
    search_space = optuna.create_study(direction ="minimize", sampler=optuna.samplers.TPESampler())
    # search_space = optuna.create_study(direction ="maximize", sampler=optuna.samplers.TPESampler())
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