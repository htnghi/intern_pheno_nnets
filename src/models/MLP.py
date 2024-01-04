
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA

import pickle as pk
import torch
from torch import optim
from torch.nn import Sequential, Linear
from torch.nn import ReLU, LeakyReLU, Tanh, Dropout 
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
def decomposition_PCA(X_train, X_val, exp_var):
    pca = PCA(exp_var)
    pca.fit(X_train)
    X_train_scaled = pca.transform(X_train)
    X_val_scaled = pca.transform(X_val)
    pk.dump(pca, open('./pca.pkl', 'wb'))
    print('Saved pca_fittor at {}, shape={}|{}'.format('./pca.pkl', X_train.shape, X_val.shape))
    
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
# Build MLP Model
# ==============================================================
def MLP(in_features, tuned_params):
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
    for i in range(tuned_params['n_layers']): 
        out_features = int(in_features * tuned_params['initial_outfeatures_factor'])
        layers.append(Linear(in_features, out_features))
        act_layer = get_activation_func(tuned_params['activation'])
        layers.append(act_layer)     
        in_features = out_features
    layers.append(Dropout(tuned_params['dropout']))
    layers.append(Linear(in_features, n_outputs))

    return Sequential(*layers)

    
# ==============================================================
# The trainning loop and testing the model with tuned params
# ==============================================================
def train_each_epoch(model, train_loader, loss_function, optimizer, device):
    arr_losses = []
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        model.train()
        pred_outputs = model(inputs)
        loss = loss_function(pred_outputs, targets)
        arr_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.average(arr_losses)

def train_loop(X_train, y_train, hyperparameters, device):

    # extract training and tuned parameters
    batch_size = 32
    num_epochs = hyperparameters['avg_epochs']
    learning_rate = hyperparameters['learning_rate']
    momentum = hyperparameters['weight_decay']
    # optimizer_type = hyperparameters['optimizer']

    # number of input features
    n_inputs = np.size(X_train, 1)

    # transform to tensor 
    tensor_X_train, tensor_y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    
    # define data loaders for training
    train_loader = DataLoader(dataset=list(zip(tensor_X_train, tensor_y_train)), batch_size=batch_size, shuffle=True) 

    # create and init the model
    model = MLP(n_inputs, hyperparameters).to(device)
    

    # define loss function
    loss_function = torch.nn.MSELoss() 

    for epoch in range(num_epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=momentum)
        avg_loss = train_each_epoch(model, train_loader, loss_function, optimizer, device)
        print ('Epoch {}/{}: avg_loss={:.5f}'.format(epoch, num_epochs, avg_loss))
    return model

# ==============================================================
# Call and train model
# ==============================================================
def run_train_MLP(datapath, X_train, y_train, X_test, y_test, hyperparameters, data_variants, device):

    # set seed
    set_seeds()
    # preprocessing data
    y_train, y_test = preprocess_mimax_scaler(y_train, y_test)

    if data_variants[1] == 1: # standard scaler
        X_train, X_test = preprocess_standard_scaler(X_train, X_test)
    elif data_variants[2] == 1: # pca fitting
        X_train, X_test = decomposition_PCA(X_train, X_test, hyperparameters['pca'])
    
    # training model
    trained_model = train_loop(X_train, y_train, hyperparameters, device)

    # save the trained model
    # torch.save(model, datapath + '/utils/tuned_MLP.model')

    # load the trained model
    # print("Loading the trained MLP model ...\n")
    # model = torch.load('datapath + '/utils/tuned_MLP.model')

    # ---------------------------------------------
    # Test the trained model with test dataset
    # ---------------------------------------------
    # pca_reloaded = pk.load(open('./pca.pkl', 'rb'))
    # X_test = pca_reloaded.transform(X_test)
    tensor_X_test = torch.Tensor(X_test)

    trained_model.eval()
    with torch.no_grad():

        y_preds_1 = trained_model(tensor_X_test)

        # change to numpy for calculating metrics in scikit learn library
        y_preds_1 = y_preds.detach().squeeze().numpy()
        y_test  = y_test.squeeze()

        if device == torch.device('cpu'):
            y_preds = y_preds_1.detach().squeeze().numpy()
        else:
            y_preds = y_preds_1.cpu().detach().squeeze().numpy()
    

        # collect mse, r2, explained variance
        test_mse = mean_squared_error(y_test, y_preds)
        test_exp_variance = explained_variance_score(y_test, y_preds)
        test_r2 = r2_score(y_test, y_preds)
        test_mae = mean_absolute_error(y_test, y_preds)

        print('--------------------------------------------------------------')
        print('Test results: avg_loss={:.4f}, avg_expvar={:.4f}, avg_r2score={:.4f}, avg_mae={:.4f}'.format(test_mse, test_exp_variance, test_r2, test_mae))
        print('--------------------------------------------------------------')

    # plot the model
    x_plot = np.arange(len(y_preds))
    plt.scatter(x_plot, y_test, alpha=0.5, label='ground_true')
    plt.plot(x_plot, y_preds, label='prediction', color='r')
    plt.xlabel('Samples', fontsize=13)
    plt.ylabel('Phenotype (g)', fontsize=13)
    plt.grid()
    plt.legend()
    # plt.show()
    
    # save to file
    minmax = '_minmax' if data_variants[0] == True else ''
    standard = '_strandard' if data_variants[1] == True else ''
    pcafitting = '_pcafitting' if data_variants[2] == True else ''
    pheno = str(data_variants[3])
    figurename = datapath + '/test_mlp_pheno' + pheno + minmax + standard + pcafitting + '.svg'
    plt.savefig(figurename, bbox_inches='tight')
    print('Saved the figure: ', figurename)

    return trained_model
