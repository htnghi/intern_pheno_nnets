import json
import copy
import torch
import optuna
import sklearn
import random

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

from torch import optim
from torch.nn import Sequential, MaxPool1d, Flatten, LeakyReLU, BatchNorm1d, Dropout, Linear, ReLU, Tanh

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

# ==============================================================
# Utils/Help function
# ==============================================================
def get_activation_func(name):
    if name == 'ReLU':
        act_func = ReLU()
    elif name == 'LeakyReLU':
        act_func = LeakyReLU()
    else:
        act_func = Tanh()
    return act_func

# min-max normalization
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
def decomposition_PCA(X_train, X_val, tuning_params):
    pca = PCA(tuning_params['pca'])
    pca.fit(X_train)
    X_train_scaled = pca.transform(X_train)
    X_val_scaled = pca.transform(X_val)
    # pk.dump(pca, open('./pca.pkl', 'wb'))
    # print('shape after PCA: train ={}, val={}'.format(X_train.shape, X_val.shape))
    return X_train_scaled, X_val_scaled

def set_seeds(seed: int=42):
    """
    Set all seeds of libs with a specific function for reproducibility of results

    :param seed: seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        # out_features = optuna_trial.suggest_int("n_units_l{}".format(i), 30, 300)
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
def train_one_epoch(model, train_loader, loss_function, optimizer, device):
    
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        pred_outputs = model(inputs)
        targets = targets.reshape((targets.shape[0], 1))
        loss_training = loss_function(pred_outputs, targets)
        optimizer.zero_grad()
        loss_training.backward()
        optimizer.step()

def validate_one_epoch(model, val_loader, loss_function, device):

    # arrays for tracking eval results
    avg_loss = 0.0
    arr_val_losses = []

    # evaluate the trained model
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            # cast the inputs and targets into float
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            arr_val_losses.append(loss.item())

    avg_loss = np.average(arr_val_losses)
    return avg_loss

def predict(model, val_loader, device):
    model.eval()
    predictions = None
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs  = inputs.float()
            outputs = model(inputs)
            # print('Target in predic function', targets)
            # print('Outputs', outputs)
            predictions = torch.clone(outputs) if predictions is None else torch.cat((predictions, outputs))
            # print('Predictions', predictions.shape)
    return predictions.detach().numpy()

def train_val_loop(model, training_params, tuning_params, X_train, y_train, X_val, y_val, device):

    # transform data to tensor format
    tensor_X_train, tensor_y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    tensor_X_val, tensor_y_val = torch.Tensor(X_val), torch.Tensor(y_val)
    
    # define data loaders for training and testing data
    train_loader = DataLoader(dataset=list(zip(tensor_X_train, tensor_y_train)), batch_size=training_params['batch_size'], shuffle=True, num_workers=2)
    val_loader   = DataLoader(dataset=list(zip(tensor_X_val, tensor_y_val)), batch_size=training_params['batch_size'], shuffle=False, num_workers=2)

    # define loss function and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=tuning_params['learning_rate'], weight_decay=tuning_params['weight_decay'])
    
    # track the best loss value and best model
    best_model = copy.deepcopy(model)
    best_loss  = None

    # track the epoch with best values
    epochs_improvement = 0
    early_stopping_point = None

    # training loop over epochs
    num_epochs = training_params['num_epochs']
    early_stop_patience = training_params['early_stop']
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, loss_function, device)
        if best_loss == None or val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            epochs_improvement = 0
        else:
            epochs_improvement += 1
        
        print('Epoch {}/{}: current_loss={:.5f} | best_loss={:.5f}'.format(epoch, num_epochs, val_loss, best_loss))

        # try to stop early
        if epoch >= 20 and epochs_improvement >= early_stop_patience:
            print("Early Stopping at epoch " + str(epoch))
            early_stopping_point = epoch - early_stop_patience
            model = best_model
            y_pred = predict(model, val_loader, device)
            return y_pred, early_stopping_point
    
    # return the best predicted values
    y_pred = predict(best_model, val_loader, device)

    return y_pred, early_stopping_point

# ==============================================================
# Define objective function for tuning hyperparameters
# ==============================================================
def objective(trial, X, y, data_variants, training_params_dict, device):

    # for tuning parameters
    tuning_params_dict = {
        'learning_rate': trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]), 
        # 'optimizer': trial.suggest_categorical('optimizer', ["Adam", "SGD"]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2),
        'initial_outfeatures_factor': trial.suggest_float('initial_outfeatures_factor', 0.05, 0.7, step=0.001),
        'activation': trial.suggest_categorical('activation', ['LeakyReLU', 'ReLU', 'Tanh']),
        'n_layers': trial.suggest_int("n_layers", 1, 5),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.05),
        'pca': trial.suggest_float('pca', 0.75, 0.95, step=0.05)
    }

    # extract preprocessed data variants for tuning
    minmax_scaler_mode = data_variants[0]
    standard_scaler_mode = data_variants[1]
    pca_fitting_mode = data_variants[2]

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

    # forl cross-validation kfolds, default = 5 folds
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

     # main loop with cv-folding
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X, y)):

        # prepare data for training and validating in each fold
        print('Fold {}: num_train_ids={}, num_val_ids={}'.format(fold, len(train_ids), len(val_ids)))
        X_train, y_train, X_val, y_val = X[train_ids], y[train_ids], X[val_ids], y[val_ids]

        # preprocessing data
        if minmax_scaler_mode == True:
            y_train, y_val = preprocess_mimax_scaler(y_train, y_val)
        if standard_scaler_mode == True:
            X_train, X_val = preprocess_standard_scaler(X_train, X_val)
        elif pca_fitting_mode == True:
            X_train, X_val = decomposition_PCA(X_train, X_val, tuning_params=tuning_params_dict)

        # create model
        num_features = X_train.shape[1]
        try:
            model = MLP(trial, in_features=num_features, tuning_params=tuning_params_dict).to(device)
    
        except Exception as err:
            print('Trial failed. Error in model creation, {}'.format(err))
            raise optuna.exceptions.TrialPruned()

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
            raise optuna.exceptions.TrialPruned()
    
    # return the average val loss
    current_val_loss = float(np.mean(first_obj_values))
    current_val_expv = float(np.mean(second_obj_values))

    # Average value of early stopping points of all innerfolds for refitting of final model
    early_stopping_point = int(np.mean(early_stopping_points))
    print('----------------------------------------------')
    print("Average early_stopping_point: {}| avg_exp_var={:.5f}| avg_loss={:.5f}".format(early_stopping_point, current_val_expv, current_val_loss))

    print('----------------------------------------------\n')

    # return current_val_expv
    return current_val_loss

# ==============================================================
# Call tuning function
# ==============================================================
def tuning_MLP(datapath, X, y, data_variants, training_params_dict, device):

    set_seeds()

    # for tracking the best validation result
    overall_results = {}

    # create an optuna tuning object, num trials default = 20
    num_trials = training_params_dict['num_trials']
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=training_params_dict['optunaseed']),
        pruner=optuna.pruners.PercentilePruner(percentile=training_params_dict['percentile'], n_min_trials=training_params_dict['min_trials'])
    )
    
    # searching loop with objective tuning
    study.optimize(lambda trial: objective(trial, X, y, data_variants, training_params_dict), n_trials=num_trials)


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

    # record best parameters to file
    minmax = '_minmax' if data_variants[0] == True else ''
    standard = '_standard' if data_variants[1] == True else ''
    pcafitting = '_pcafitting' if data_variants[2] == True else ''
    pheno = str(data_variants[3])
    with open(f"./tuned_mlp_" + "pheno" + pheno + minmax + standard + pcafitting + ".json", 'w') as fp:
        json.dump(best_params, fp)

    # fig_optim_history = optuna.visualization.plot_optimization_history(study)
    # fig_inter_values = optuna.visualization.plot_intermediate_values(study)
    # fig_optim_history.write_image("./optimhisto_mlp_pheno" + pheno + minmax + standard + pcafitting + ".pdf")
    # fig_inter_values.write_image("./intervalue_mlp_pheno" + pheno + minmax + standard + pcafitting + ".pdf")
    # fig_optim_history.show()
    # fig_inter_values.show()

    return overall_results