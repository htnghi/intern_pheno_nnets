import json
import copy
import torch
import optuna
import sklearn

import numpy as np
import pandas as pd

from pandas import read_csv

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

from torch import optim

from torch.nn import Sequential, MaxPool1d, Flatten, LeakyReLU, BatchNorm1d, Dropout, Linear, ReLU, Tanh

from torch.optim import SGD, Adam

from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data import random_split

# ==============================================================
# Utils/Help function
# ==============================================================
def get_activation_func(name):
    act_func = Tanh()
    if name == 'relu':
        act_func = ReLU()
    elif name == 'leakyrelu':
        act_func = LeakyReLU()
    return act_func

# ==============================================================
# Define MLP Model
# ==============================================================
def MLP(optuna_trial, in_features, tuning_params):
    """
    Generate sequential network model with optuna optimization.

    :param optuna_trial: optuna trial class
    :param in_features: num of input nodes
    :param n_layers: num of hidden layers
    :param initial_outfeatures_factor: init the number of out features
    :param dropout: perc of final layer dropout
    :param activation: name of activation function
    :param n_output: num of output nodes
    :return: sequential multi layer perceptron model 
    """
    n_outputs = 1
    layers = []
    for i in range(tuning_params['n_layers']): 
        out_features = int(in_features * tuning_params['initial_outfeatures_factor'])
        layers.append(Linear(in_features, out_features))
        act_layer = get_activation_func(tuning_params['activation'])
        layers.append(act_layer)     
        in_features = out_features
    layers.append(Dropout(tuning_params['dropout']))
    layers.append(Linear(in_features, n_outputs))

    return Sequential(*layers)

# ==============================================================
# Define training and validation loop
# ==============================================================
def train_one_epoch(model, train_loader, loss_function, optimizer):

    for i, (inputs, targets) in enumerate(train_loader):
        model.train()
        pred_outputs = model(inputs)
        targets = targets.reshape((targets.shape[0], 1))
        loss_training = loss_function(pred_outputs, targets)
        optimizer.zero_grad()
        loss_training.backward()
        optimizer.step()

def validate_one_epoch(model, val_loader, loss_function):

    # arrays for tracking eval results
    avg_loss = 0.0
    arr_val_losses = []

    # evaluate the trained model
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            # cast the inputs and targets into float
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            arr_val_losses.append(loss.item())

    avg_loss = np.average(arr_val_losses)
    return avg_loss

def predict(model, val_loader):
    model.eval()
    predictions = None
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs  = inputs.float()
            outputs = model(inputs)
            predictions = torch.clone(outputs) if predictions is None else torch.cat((predictions, outputs))
    return predictions.detach().numpy()

def train_val_loop(model, training_params, tuning_params, X_train, y_train, X_val, y_val):

    # transform data to tensor format
    tensor_X_train, tensor_y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    tensor_X_val, tensor_y_val = torch.Tensor(X_val), torch.Tensor(y_val)

    # define data loaders for training and testing data
    train_loader = DataLoader(dataset=list(zip(tensor_X_train, tensor_y_train)), batch_size=training_params['batch_size'], shuffle=True)
    val_loader   = DataLoader(dataset=list(zip(tensor_X_val, tensor_y_val)), batch_size=training_params['batch_size'], shuffle=True)

    # define loss function and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = getattr(optim, tuning_params['optimizer'])(model.parameters(),
                    lr=tuning_params['learning_rate'], weight_decay=tuning_params['weight_decay'])
    
    # track the best loss value and best model
    best_model = copy.deepcopy(model)
    best_loss  = None

    # track the epoch with best values
    epochs_improvement = 0
    early_stopping_point = None

    # training loop over epochs
    num_epochs = training_params['n_epochs']
    early_stop_patience = training_params['early_stop_patience']
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, loss_function, optimizer)
        val_loss = validate_one_epoch(model, val_loader, loss_function)
        if best_loss == None or val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            epochs_improvement = 0
        else:
            epochs_improvement += 1
        
        print('Epoch ' + str(epoch) + ' of ' + str(num_epochs))
        print('Current val_loss=' + str(val_loss) + ', best val_loss=' + str(best_loss))

        # try to stop early
        # if epoch >= 20 and epochs_improvement >= tuning_params['early_stop_patience']:
        #     print("Early Stopping at epoch " + str(epoch))
        #     early_stopping_point = epoch - tuning_params['early_stop_patience']
        #     model = best_model
        #     return predict(model, val_loader), early_stopping_point
        
        if epoch >= 25 and epochs_improvement >= early_stop_patience:
            print("Early Stopping at epoch " + str(epoch))
            early_stopping_point = epoch - early_stop_patience
            model = best_model
            return predict(model, val_loader), early_stopping_point
    
    return predict(best_model, val_loader), early_stopping_point

# ==============================================================
# Define objective function for tuning hyperparameters
# ==============================================================
def objective(trial, X, y):

    # for extracting related parameters of training
    training_params_dict = {}
    training_params_dict['batch_size'] = 32
    training_params_dict['n_epochs']   = 200
    training_params_dict['width_onehot'] = 4
    training_params_dict['early_stop_patience'] = 25

    # for tuning parameters
    tuning_params_dict = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1), 
        'optimizer': trial.suggest_categorical('optimizer', ["Adam", "SGD"]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-2),
        'initial_outfeatures_factor': trial.suggest_float('initial_outfeatures_factor', 0.01, 0.8, step=0.01),
        'activation': trial.suggest_categorical('activation', ['leakyrelu', 'relu', 'tanh']),
        'n_layers': trial.suggest_int("n_layers", 1, 5),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.05),
        # 'early_stop_patience': trial.suggest_int("early_stop_patience", 0, 6, step=2),
        'pca': trial.suggest_float('pca', 0.7, 0.95, step=0.05)
    }

    # log early stopping point at each fold
    early_stopping_points = []

    # create model
    num_features = X.shape[1]
    try:
        model = MLP(trial, in_features=num_features, tuning_params=tuning_params_dict)
    except Exception as err:
        print('Trial failed. Error in model creation, {}'.format(err))
        raise optuna.exceptions.TrialPruned()
    
    # iterate for training and tuning
    print("Params for Trial " + str(trial.number))
    print(trial.params)

    # tracking the results
    objective_values = []

    # forl cross-validation kfolds, default = 5 folds
    kfold = KFold(n_splits=5, shuffle=True)

     # main loop with cv-folding
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X, y)):

        # prepare data for training and validating in each fold
        print('Fold {}: len(train_ids)={:5d}, len(val_ids)={:5d}'.format(fold, len(train_ids), len(val_ids)))
        X_train, y_train, X_val, y_val = X[train_ids], y[train_ids], X[val_ids], y[val_ids]

        # call training model over each fold
        try:
            y_pred, stopping_point = train_val_loop(model, training_params_dict, tuning_params_dict,
                                     X_train, y_train, X_val, y_val)
            
            # record the early-stopping points
            if stopping_point is not None:
                early_stopping_points.append(stopping_point)
            else:
                early_stopping_points.append(training_params_dict['n_epochs'])
            
            # calculate objective value
            obj_value = sklearn.metrics.mean_squared_error(y_true=y_val, y_pred=y_pred)

            # report pruned values
            trial.report(value=obj_value, step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            # accumulate the obj val losses
            objective_values.append(obj_value)

        # for pruning the tuning process
        except (RuntimeError, TypeError, ValueError) as exc:
            print(exc)
            if 'out of memory' in str(exc):
                print('Out of memory')
            else:
                print('Trial failed. Error in optim loop.')
            raise optuna.exceptions.TrialPruned()
    
    # return the average val loss
    current_val_result = float(np.mean(objective_values))

    # Average value of early stopping points of all innerfolds for refitting of final model
    early_stopping_point = int(np.mean(early_stopping_points))
    print("Average early_stopping_point:", early_stopping_point)


    print('----------------------------------------------\n')

    return current_val_result

# ==============================================================
# Call tuning function
# ==============================================================
def tuning_MLP(datapath, X, y):

    # for tracking the best validation result
    best_val_result = None
    overall_results = {}

    # create an optuna tuning object, num trials default = 20
    num_trials = 120
    study = optuna.create_study(
        direction ="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.PercentilePruner(percentile=80, n_min_trials=20)
    )
    
    # searching loop with objective tuning
    study.optimize(lambda trial: objective(trial, X, y), n_trials=num_trials)


    # print statistics after tuning
    print("Optuna study finished, study statistics:")
    print("  Finished trials: ", len(study.trials))
    print("  Pruned trials: ", len(study.get_trials(states=(optuna.trial.TrialState.PRUNED,))))
    print("  Completed trials: ", len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))))
    print("  Best Trial: ", study.best_trial.number)
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print("    {}: {}".format(key, value))

    print('----------------------------------------------\n')

    best_params = study.best_trial.params
    overall_results[key] = {'best_params': best_params}

    
    # model_params = study.best_trial.params
    # model_params['optuna_best_trial_number'] =  study.best_trial.number 
    # model_params['optuna_best_trial_value'] = float(np.round(study.best_value, 6))
    # model_params["n_trials"] = num_trials

    with open(f"./tuning_mlp_num_trials_" + str(num_trials) + ".json", 'w') as fp:
        json.dump(best_params, fp)

    return overall_results