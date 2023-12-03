import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

import torch
from torch import optim
from torch.nn import Linear
from torch.nn import ReLU, LeakyReLU, Tanh
from torch.nn import Module, Sequential
from torch.nn import MaxPool1d, Conv1d, BatchNorm1d, Flatten, Dropout
from torch.utils.data import DataLoader

import optuna
import json
import time

# ==============================================================
# Define CNN Model
# ==============================================================

def parse_activation_func(name):
    if name == 'ReLU':
        act_func = ReLU()
    elif name == 'LeakyReLU' :
        act_func = LeakyReLU()
    else:
        act_func = Tanh()
    return act_func

def CNN(optuna_trial, num_features, kernel_size, stride_percentage, n_layers, factor_out_linear_features, dropout, activation1, activation2):
    """
    Generate sequential network model with optuna optimization.

    :param optuna_trial: optuna trial class with other tunning parameters
    :param num_features: num of input features from the original dataset
    :param n_layers: num of conv hidden layers
    :param dropout: perc of final layer dropout
    :param activation: type of activation functions
    :return: sequential multi layer perceptron model
    """

    # print('DBG: num features of input dataset - {:4d}'.format(num_features))

    # for the first conv layer
    # in_filters = 1 # for additive encoding
    in_filters = 4 # for one hot encoding
    out_filters = 2**2
    layers = []

    # for activation functions, ReLU, LeakyReLU, Tanh
    act_func1 = parse_activation_func(activation1)
    act_func2 = parse_activation_func(activation2)

    for i in range(n_layers):

        # print('DBG: conv layer {:2d}, in={:4d}, out={:4d}'.format(i, in_filters, out_filters))
        stride = max(1, int(kernel_size*stride_percentage))

        layers.append(Conv1d(in_filters, out_filters, kernel_size=kernel_size, stride=stride))
        layers.append(act_func1)
        layers.append(BatchNorm1d(out_filters))
        layers.append(Dropout(dropout))

        # update in an out filters
        in_filters = out_filters
        out_filters = out_filters * 2

    layers.append(MaxPool1d(kernel_size))
    layers.append(Flatten())

    # check number of inputs and outputs before going to liner layers
    # in_linear_features = Sequential(*layers)(torch.zeros(size=(50, 1, num_features))).shape[1] # for additive encoding
    in_linear_features = Sequential(*layers)(torch.zeros(size=(50, 4, num_features))).shape[1] # for one hot encoding
    out_linear_features = int(in_linear_features * factor_out_linear_features)
    # print('DBG: linear layer, in_features={}, out_features={:4d}'.format(in_linear_features, out_linear_features))

    # linear layers
    layers.append(Linear(in_linear_features, out_linear_features))
    layers.append(act_func2)
    layers.append(BatchNorm1d(out_linear_features))
    layers.append(Dropout(dropout))
    layers.append(Linear(in_features=out_linear_features, out_features=1))
    print("--------------------------------------------------------------------")

    return Sequential(*layers)

# ==============================================================
# The trainning loop
# ==============================================================
def train_model(num_epochs, X, y, batch_size, params, optuna_trial):

    # start time measurement for training
    # start = time.time()

    # declare arrays for storing total loss and other metrics
    arr_val_losses  = []
    arr_r2_scores = []
    arr_exp_vars = []

    # repare and split datasets
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, shuffle=True)

    # MinMax Scaler
    minmax_scaler = MinMaxScaler()
    y_train = np.expand_dims(y_train, axis=1)
    y_val = np.expand_dims(y_val, axis=1)
    y_train_scaled = minmax_scaler.fit_transform(y_train)
    y_val_scaled   = minmax_scaler.fit_transform(y_val)

    # normalize dataset using StandardScaler
    # standard_scaler = StandardScaler()
    # standard_scaler.fit(X_train)
    # X_train = standard_scaler.transform(X_train)
    # X_val   = standard_scaler.transform(X_val)

    # PCA
    # pca = PCA(params['pca'])
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    # X_val = pca.transform(X_val)
    # pk.dump(pca, open('./pca.pkl', 'wb'))
    # print('shape after PCA: train ={}, val={}'.format(X_train.shape, X_val.shape))


    # transform to tensor 
    tensor_X_train, tensor_y_train = torch.Tensor(X_train), torch.Tensor(y_train_scaled)
    tensor_X_val, tensor_y_val = torch.Tensor(X_val), torch.Tensor(y_val_scaled)

    # squeeze y for training CNN to tensor
    tensor_y_train, tensor_y_val = tensor_y_train.view(len(y_train),1), tensor_y_val.view(len(y_val),1)

    # Reshape X (specified for each encoding)
    # tensor_X_train, tensor_X_val  = tensor_X_train.unsqueeze(1), tensor_X_val.unsqueeze(1) # for additive encoding
    tensor_X_train, tensor_X_val = torch.swapaxes(tensor_X_train, 1, 2), torch.swapaxes(tensor_X_val, 1, 2) #for onehot

    # format all data with DataLoader
    train_loader = DataLoader(dataset = list(zip(tensor_X_train, tensor_y_train)), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader  = DataLoader(dataset = list(zip(tensor_X_val, tensor_y_val)), batch_size=int(batch_size/2), shuffle=True, num_workers=0)

    # declare the MLP model
    cnn_model = CNN(optuna_trial,
            num_features = 10000,
            kernel_size = params['kernel_size'],
            stride_percentage = params ['stride_percentage'],
            n_layers = params['n_layers'],
            factor_out_linear_features= params['factor_out_linear_features'],
            dropout = params['dropout'],
            activation1 = params['activation1'],
            activation2 = params['activation2']
            )

    # define loss function and optimizer
    loss_function = torch.nn.MSELoss()   
    optimizer = getattr(optim, params['optimizer'])(cnn_model.parameters(), lr= params['learning_rate'], weight_decay=params['weight_decay'])

    for epoch in range(num_epochs):
        
        # iterate through training data loader
        for i, (inputs, targets) in enumerate(train_loader):
            
            # cast the inputs and targets into float
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))

            cnn_model.train()
            outputs = cnn_model(inputs)
            loss = loss_function(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # model evaluation in each epoch
        epoch_val_losses = []
        epoch_val_expvars = []
        epoch_val_r2scors = []

        cnn_model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):

                # cast the inputs and targets into float
                inputs, targets = inputs.float(), targets.float()

                # make sure the targets reshaped and perform forward
                targets = targets.reshape((targets.shape[0], 1))

                val_outputs = cnn_model(inputs)

                # calculate the loss
                val_loss = loss_function(val_outputs, targets)
                epoch_val_losses.append(val_loss.item())

                # calculate the exp_var to check during optuna tuning
                np_targets = targets.squeeze().numpy()
                np_predics = val_outputs.detach().squeeze().numpy()
                
                test_expvar = explained_variance_score(np_targets, np_predics)
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

# ==========================================================
# Objective function for tuning hyper-parameters
# ==========================================================
def objective(X, y, optuna_trial):
    """
    Objective function to run bayesian hyperparameter tuning

    :param X: dataset X for features
    :param y: dataset y for labels
    :param optuna_trial: optuna study along with the external variables, num_layers and num_filters
    :return: mean RMSE test loss
    """

    print("")

    # for tuning samples 
    params = {
              'learning_rate': optuna_trial.suggest_float('learning_rate', 1e-6, 1e-2), 
              'optimizer': optuna_trial.suggest_categorical('optimizer', ["Adam", "RMSprop", "SGD"]),
              'weight_decay': optuna_trial.suggest_float('weight_decay', 1e-4, 1e-2),
              'kernel_size': optuna_trial.suggest_int("kernel_size", 2, 8),
              'stride_percentage': optuna_trial.suggest_float('stride_percentage', 0.1, 1.0, step=0.1),
              'n_layers': optuna_trial.suggest_int("n_layers", 1, 4),
              'factor_out_linear_features': optuna_trial.suggest_float('factor_out_linear_features', 0.5, 1, step=0.1),
              'activation1': optuna_trial.suggest_categorical('activation1', ['ReLU', 'LeakyReLU', 'Tanh']),
              'activation2': optuna_trial.suggest_categorical('activation2', ['ReLU', 'LeakyReLU', 'Tanh']),
              'dropout': optuna_trial.suggest_float('dropout', 0.1, 0.5)
              }
    
    # define external values to tune the hyperparameters
    # num_layers = optuna_trial.suggest_int("num_layers", 1, 3)
    # num_filters = [int(optuna_trial.suggest_discrete_uniform("num_filter_"+str(i), 128, 256, 128)) for i in range(num_layers)]

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
def trial_train_and_tune_CNN(datapath, X, y):

    # number of trials
    num_trials = 20

    search_space = optuna.create_study(direction ="minimize", sampler=optuna.samplers.TPESampler())
    # search_space = optuna.create_study(direction ="maximize", sampler=optuna.samplers.TPESampler())
    search_space.optimize(lambda trial: objective(X, y, trial), n_trials=num_trials)

    model_params = search_space.best_trial.params
    model_params['optuna_best_trial_number'] =  search_space.best_trial.number 
    model_params['optuna_best_trial_value'] = float(np.round(search_space.best_value, 6))
    model_params["n_trials"] = num_trials

    with open(f"./tuning_mlp_model_with_optuna_num_trials_" + str(num_trials) + ".json", 'w') as fp: 
        json.dump(model_params, fp)
    
    # print()
    # print('----------------------------------------------------')
    # print("Tuning CNN model with Optuna: ")
    # print("The result is writen at ./tuning/" + "tuning_cnn_model_" + str(num_trials) + ".json")
    # print('----------------------------------------------------\n')

    return 0