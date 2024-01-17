import json
import copy
import torch
import optuna
import sklearn
import random
import joblib
import pathlib

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

from torch import optim
from torch.nn import MaxPool1d, Conv1d
from torch.nn import Sequential, MaxPool1d, Flatten, LeakyReLU, BatchNorm1d, Dropout, Linear, ReLU, Tanh

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

# ==============================================================
# Utils/Help function
# ==============================================================
def get_activation_func(name):
    if name == 'ReLU':
        act_func = ReLU()
    elif name == 'LeakyReLU' :
        act_func = LeakyReLU()
    else:
        act_func = Tanh()
    return act_func

# MinMax Scaler
def preprocess_mimax_scaler(y_train, y_val):
    minmax_scaler = MinMaxScaler()
    y_train = np.expand_dims(y_train, axis=1)
    y_val = np.expand_dims(y_val, axis=1)
    y_train_scaled = minmax_scaler.fit_transform(y_train)
    y_val_scaled   = minmax_scaler.transform(y_val)
    return y_train_scaled, y_val_scaled

# normalize dataset using StandardScaler
def preprocess_standard_scaler(X_train, X_val):
    standard_scaler = StandardScaler()
    standard_scaler.fit(X_train)
    X_train_scaled = standard_scaler.transform(X_train)
    X_val_scaled   = standard_scaler.transform(X_val)
    return X_train_scaled, X_val_scaled

# decomposition PCA
def decomposition_PCA(X_train, X_val, pca_value):
    pca = PCA(pca_value)
    pca.fit(X_train)
    X_train_scaled = pca.transform(X_train)
    X_val_scaled = pca.transform(X_val)
    return X_train_scaled, X_val_scaled

# set seeds for reproducibility
def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # if gpu cuda available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

# ==============================================================
# Define CNN Model
# ==============================================================
def CNN1D(num_features, tuning_params):
    """
    CNN model with hyperparameter tuning by optuna optimization.

    :param n_layers: the number of convolutional layers
    :param kernel_size: the size of convolutional kernel
    :param stride: the ratio of stride to kernel size
    :param out_factor: the ratio of the number of nodes to the number of input features after convolutional layers
    :param dropout: dropout rate
    :param activation: activation function
    :return: sequential multi layer perceptron model
    """

    # for the first conv layer
    in_filters = 4 # for one hot encoding
    out_filters = 2**2
    layers = []

    # for activation functions, ReLU, LeakyReLU, Tanh
    act_func1 = get_activation_func(tuning_params['activation1'])
    act_func2 = get_activation_func(tuning_params['activation2'])

    for i in range(tuning_params['n_layers']):

        stride = max(1, int(tuning_params['kernel_size'] * tuning_params['stride']))
        layers.append(Conv1d(in_filters, out_filters, kernel_size=tuning_params['kernel_size'], stride=stride))
        layers.append(act_func1)
        layers.append(BatchNorm1d(out_filters))
        layers.append(Dropout(tuning_params['dropout']))

        # update in an out filters
        in_filters = out_filters
        out_filters = out_filters * 2

    layers.append(MaxPool1d(tuning_params['kernel_size']))
    layers.append(Flatten())

    # check number of inputs and outputs before going to liner layers
    in_linear_features = Sequential(*layers)(torch.zeros(size=(50, 4, num_features))).shape[1] # for one hot encoding
    out_linear_features = int(in_linear_features * tuning_params['out_factor'])
    # print('DBG: linear layer, in_features={}, out_features={:4d}'.format(in_linear_features, out_linear_features))

    # linear layers
    layers.append(Linear(in_linear_features, out_linear_features))
    layers.append(act_func2)
    layers.append(BatchNorm1d(out_linear_features))
    layers.append(Dropout(tuning_params['dropout']))
    layers.append(Linear(in_features=out_linear_features, out_features=1))
    print("--------------------------------------------------------------------")

    return Sequential(*layers)

# ==============================================================
# Define training and validation loop
# ==============================================================

# Function to train the model for one epoch
def train_one_epoch(model, train_loader, loss_function, optimizer, device):

    model.train()
    # iterate through the train loader
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # forward pass 
        pred_outputs = model(inputs)
        # calculate training loss
        loss_training = loss_function(pred_outputs, targets)
        # backward pass and optimization
        optimizer.zero_grad()
        loss_training.backward()
        optimizer.step()

# Function to validate the model for one epoch
def validate_one_epoch(model, val_loader, loss_function, device):

    # arrays for tracking eval results
    avg_loss = 0.0
    arr_val_losses = []

    # evaluate the trained model
    model.eval()
    with torch.no_grad():
        # Iterate through the validation loader
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            arr_val_losses.append(loss.item())
    
    # calculate average validation loss
    avg_loss = np.average(arr_val_losses)
    return avg_loss

# Function to make predictions using the given model
def predict(model, val_loader, device):
    model.eval()
    predictions = None
    with torch.no_grad():
        # iterate through the validation loader
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs  = inputs.float()
            outputs = model(inputs)
            # concatenate the predictions
            predictions = torch.clone(outputs) if predictions is None else torch.cat((predictions, outputs))
    
    # convert predictions to numpy array based on device
    if device == torch.device('cpu'):
        ret_output = predictions.detach().numpy()
    else:
        ret_output = predictions.cpu().detach().numpy()
    
    return ret_output

# Function to train model on train loader and evaluate on validation loader
# also return early_stopping_point
def train_val_loop(model, training_params, tuning_params, X_train, y_train, X_val, y_val, device):

    # transform data to tensor format
    tensor_X_train, tensor_y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    tensor_X_val, tensor_y_val = torch.Tensor(X_val), torch.Tensor(y_val)

    # squeeze y to get suitable y dims for training CNN
    tensor_y_train, tensor_y_val = tensor_y_train.view(len(y_train),1), tensor_y_val.view(len(y_val),1)

    # Reshape X (specified for one-hot encoding)
    tensor_X_train, tensor_X_val = torch.swapaxes(tensor_X_train, 1, 2), torch.swapaxes(tensor_X_val, 1, 2) 

    # define data loaders for training and validaion data
    train_loader = DataLoader(dataset=list(zip(tensor_X_train, tensor_y_train)), batch_size=training_params['batch_size'], shuffle=True)
    val_loader   = DataLoader(dataset=list(zip(tensor_X_val, tensor_y_val)), batch_size=training_params['batch_size'], shuffle=False)

    # define loss function and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=tuning_params['learning_rate'], weight_decay=tuning_params['weight_decay'])
    
    # track the best loss value and best model
    best_model = copy.deepcopy(model)
    best_loss  = None

    # track the epoch with best values
    epochs_no_improvement = 0
    early_stopping_point = None

    # training loop over epochs
    num_epochs = training_params['num_epochs']
    early_stop_patience = training_params['early_stop']
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, loss_function, device)

        # check if the current validation loss is the best observed so far
        # if current val loss is not the best, increase the count of epochs with no improvement
        if best_loss == None or val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
        
        print('Epoch {}/{}: current_loss={:.5f} | best_loss={:.5f}'.format(epoch, num_epochs, val_loss, best_loss))
        
        # check if early stopping criteria are met
        # if the current epoch is greater than or equal to 20 
        # and epochs with no improvement surpass early stopping patience
        if epoch >= 20 and epochs_no_improvement >= early_stop_patience:
            # set the early stopping point
            early_stopping_point = epoch - early_stop_patience
            print("Stopped at epoch " + str(epoch) + "| " + "Early stopping point = " + str(early_stopping_point))
            # predict using the best model
            model = best_model
            y_pred = predict(model, val_loader, device)
            return y_pred, early_stopping_point
    
    # return the best predicted values
    y_pred = predict(best_model, val_loader, device)

    return y_pred, early_stopping_point

# ==============================================================
# Define objective function for tuning hyperparameters
# ==============================================================
def objective(trial, X, y, data_variants, training_params_dict, avg_stop_epochs, device):

    # for tuning parameters
    tuning_params_dict = {
        'learning_rate': trial.suggest_categorical('learning_rate', [1e-5,1e-4,1e-3,1e-2,1e-1]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-10, 1e-2),
        'kernel_size': trial.suggest_int("kernel_size", 2, 8),
        'stride': trial.suggest_float('stride', 0.1, 1.0, step=0.1),
        'n_layers': trial.suggest_int("n_layers", 1, 4),
        'out_factor': trial.suggest_float('out_factor', 0.2, 1, step=0.1),
        'activation1': trial.suggest_categorical('activation1', ['ReLU', 'LeakyReLU', 'Tanh']),
        'activation2': trial.suggest_categorical('activation2', ['ReLU', 'LeakyReLU', 'Tanh']),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    }

    # extract preprocessed data variants for tuning
    minmax_scaler_mode = data_variants[0]

    # log early stopping point at each fold
    early_stopping_points = []

    # iterate for training and tuning
    print('\n----------------------------------------------')
    print("Params for Trial " + str(trial.number))
    print(trial.params)
    print('----------------------------------------------')

    # tracking the results
    first_obj_values = []
    second_obj_values = []

    # create model
    num_features = X.shape[1]
    try:
        model = CNN1D(num_features=num_features, tuning_params=tuning_params_dict).to(device)
    except Exception as err:
        print('Trial failed. Error in model creation, {}'.format(err))
        raise optuna.exceptions.TrialPruned()

    # forl cross-validation kfolds, default = 5 folds
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

     # main loop with cv-folding
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X, y)):

        # prepare data for training and validating in each fold
        print('Fold {}: num_train_ids={}, num_val_ids={}'.format(fold, len(train_ids), len(val_ids)))
        X_train, y_train, X_val, y_val = X[train_ids], y[train_ids], X[val_ids], y[val_ids]

        # preprocessing data
        if minmax_scaler_mode == 1: # minmax scaler
            y_train, y_val = preprocess_mimax_scaler(y_train, y_val)

        # call training model over each fold
        try:
            y_pred, stopping_point = train_val_loop(model, training_params_dict, tuning_params_dict,
                                     X_train, y_train, X_val, y_val, device)
            
            # record the early-stopping points
            if stopping_point is not None:
                early_stopping_points.append(stopping_point)
            else:
                early_stopping_points.append(training_params_dict['num_epochs'])

            # calculate objective value
            obj_value1 = sklearn.metrics.mean_squared_error(y_true=y_val, y_pred=y_pred)
            obj_value2 = sklearn.metrics.explained_variance_score(y_true=y_val, y_pred=y_pred)
            print('      explained_var={:.5f} | mse_loss={:.5f}'.format(obj_value2, obj_value1))

            # report pruned values
            trial.report(value=obj_value1, step=fold)
            if trial.should_prune():
                # clean unfitted models and tuned parameters that are pruned
                # print('Clean unfitted models pruned: {}'.format(pathname+model_name))
                # if pathlib.Path(pathname+model_name).exists():
                #     pathlib.Path(pathname+model_name).unlink()
                raise optuna.exceptions.TrialPruned()
            
            # accumulate the obj val losses
            first_obj_values.append(obj_value1)
            second_obj_values.append(obj_value2)

        # for pruning the tuning process
        except (RuntimeError, TypeError, ValueError) as exc:
            print(exc)
            if 'out of memory' in str(exc):
                print('Out of memory')
            else:
                print('Trial failed. Error in optim loop.')
            
            # print('Clean unfitted models pruned: {}'.format(pathname+model_name))
            # if pathlib.Path(pathname+model_name).exists():
            #         pathlib.Path(pathname+model_name).unlink()

            raise optuna.exceptions.TrialPruned()
    
    # return the average val loss
    current_val_loss = float(np.mean(first_obj_values))
    current_val_expv = float(np.mean(second_obj_values))

    # Average value of early stopping points of all innerfolds for refitting of final model
    early_stopping_point = int(np.mean(early_stopping_points))
    print('----------------------------------------------')
    print("Average early_stopping_point: {}| avg_exp_var={:.5f}| avg_loss={:.5f}".format(early_stopping_point, current_val_expv, current_val_loss))
    print('----------------------------------------------\n')

    # try to return avg stop epochs for a specific trial 
    avg_stop_epochs[trial.number] = early_stopping_point

    return current_val_loss

# ==============================================================
# Call tuning function
# ==============================================================
def tuning_CNN(datapath, X, y, data_variants, training_params_dict, device):

    # set seeds for reproducibility
    set_seeds()

    # for tracking the tuning information
    minmax = '_minmax' if data_variants[0] == True else ''
    standard = '_standard' if data_variants[1] == True else ''
    pcafitting = '_pca' if data_variants[2] == True else ''
    pheno = str(data_variants[3])

    # create a list to keep track all stopping epochs during a hyperparameter search
    # because in some cases, the trial is pruned so the stopping epochs default = num_epochs
    avg_stopping_epochs = [training_params_dict['num_epochs']] * training_params_dict['num_trials']

    # create an optuna tuning object, num trials default = 100
    num_trials = training_params_dict['num_trials']
    study = optuna.create_study(
        study_name='cnn_'+'mseloss_'+'data'+pheno+minmax+standard+pcafitting,
        direction ="minimize",
        sampler=optuna.samplers.TPESampler(seed=training_params_dict['optunaseed']),
        pruner=optuna.pruners.PercentilePruner(percentile=training_params_dict['percentile'], n_min_trials=training_params_dict['min_trials'])
    )
    
    # searching loop with objective tuning
    study.optimize(lambda trial: objective(trial, X, y, data_variants, training_params_dict, avg_stopping_epochs, device), n_trials=num_trials)

    # gget early stopping of the best trial
    num_avg_stop_epochs = avg_stopping_epochs[study.best_trial.number]

    # print statistics after tuning
    print("Optuna study finished, study statistics:")
    print("  Finished trials: ", len(study.trials))
    print("  Pruned trials: ", len(study.get_trials(states=(optuna.trial.TrialState.PRUNED,))))
    print("  Completed trials: ", len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))))
    print("  Best Trial: ", study.best_trial.number)
    print("  Value: ", study.best_trial.value)
    print("  AVG stopping: ", num_avg_stop_epochs)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print("    {}: {}".format(key, value))

    print('----------------------------------------------\n')

    best_params = study.best_trial.params
    best_params['avg_epochs'] = num_avg_stop_epochs
    print('Check best params: {}'.format(best_params))

    # record best parameters to file
    with open(f"./tuned_cnn_" + "pheno" + pheno + minmax + standard + pcafitting + ".json", 'w') as fp:
        json.dump(best_params, fp)

    return best_params

# ==============================================================
# Evaluation the performance on test set
# ==============================================================
def evaluate_result_CNN(datapath, X_train, y_train, X_test, y_test, best_params, data_variants, device):

    # set seeds for reproducibility
    set_seeds()
    
     # for tracking the tuning information
    minmax = '_minmax' if data_variants[0] == True else ''
    standard = '_standard' if data_variants[1] == True else ''
    pcafitting = '_pca' if data_variants[2] == True else ''
    pheno = str(data_variants[3])

    # extract preprocessed data variants for tuning
    minmax_scaler_mode = data_variants[0]

    # preprocessing data
    if minmax_scaler_mode == 1: # minmax scaler
        y_train, y_test = preprocess_mimax_scaler(y_train, y_test)

    # extract training and tuned parameters
    batch_size = 32
    num_epochs = best_params['avg_epochs']
    learning_rate = best_params['learning_rate']
    momentum = best_params['weight_decay']

    # number of input features
    num_features = X_train.shape[1]

    # create model
    model = CNN1D(num_features=num_features, tuning_params=best_params).to(device)

    # transform data to tensor format
    tensor_X_train, tensor_y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    tensor_X_test, tensor_y_test = torch.Tensor(X_test), torch.Tensor(y_test)

    # squeeze y for training CNN to tensor
    tensor_y_train, tensor_y_test = tensor_y_train.view(len(y_train),1), tensor_y_test.view(len(y_test),1)

    # Reshape X (specified for one hot encoding)
    tensor_X_train, tensor_X_test = torch.swapaxes(tensor_X_train, 1, 2), torch.swapaxes(tensor_X_test, 1, 2)

    # define data loaders for training and testing data
    train_loader = DataLoader(dataset=list(zip(tensor_X_train, tensor_y_train)), batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(dataset=list(zip(tensor_X_test, tensor_y_test)), batch_size=batch_size, shuffle=False)

    # define loss function and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=momentum)

    # training loop over epochs
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, loss_function, optimizer, device)
    
    # predict result test 
    y_pred = predict(model, test_loader, device)

    # collect mse, r2, explained variance
    test_mse = sklearn.metrics.mean_squared_error(y_true=y_test, y_pred=y_pred)
    test_exp_variance = sklearn.metrics.explained_variance_score(y_true=y_test, y_pred=y_pred)
    test_r2 = sklearn.metrics.r2_score(y_true=y_test, y_pred=y_pred)
    test_mae = sklearn.metrics.mean_absolute_error(y_true=y_test, y_pred=y_pred)

    print('--------------------------------------------------------------')
    print('Test CNN results: avg_loss={:.4f}, avg_expvar={:.4f}, avg_r2score={:.4f}, avg_mae={:.4f}'.format(test_mse, test_exp_variance, test_r2, test_mae))
    print('--------------------------------------------------------------')

    return test_exp_variance

