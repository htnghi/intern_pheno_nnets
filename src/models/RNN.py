import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error, mean_absolute_error

import pickle as pk
import torch

from torch import optim
from torch.nn import LSTM, Linear
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

# minMax Scaler
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

# Decomposition PCA
def decomposition_PCA(X_train, X_val, tuning_params):
    pca = PCA(tuning_params['pca'])
    pca.fit(X_train)
    X_train_scaled = pca.transform(X_train)
    X_val_scaled = pca.transform(X_val)
    # pk.dump(pca, open('./pca.pkl', 'wb'))
    # print('shape after PCA: train ={}, val={}'.format(X_train.shape, X_val.shape))
    return X_train_scaled, X_val_scaled

# Get output after LSTM layer
class Output_lstm(torch.nn.Module):
    def __init__(self):
        super(Output_lstm, self).__init__()

    def forward(self, x):
        lstm_out, (hn, cn) = x
        return lstm_out

# Reshape data before going to linear layer
class Reshape_to_linear(torch.nn.Module):
    def __init__(self):
        super(Reshape_to_linear, self).__init__()

    def forward(self, lstm_out):
        return lstm_out[:, -1, :]
    
# ==============================================================
# Define CNN Model
# ==============================================================
def RNN(tuned_params):
    """
    Generate sequential network model with optuna optimization.

    :param optuna_trial: optuna trial class with other tunning parameters
    :param n_layers: num of rnn hidden layers
    :param hidden_size: the size of hidden state
    :param dropout: perc of final layer dropout
    :return: sequential multi layer perceptron model
    """
    
    layers = []
    n_feature = 4 # for onehot
    if tuned_params['n_layers'] > 1:
        layers.append(LSTM(input_size=n_feature, hidden_size=tuned_params['hidden_size'],
                                num_layers=tuned_params['n_layers'], dropout=tuned_params['dropout']))
    else:
        layers.append(LSTM(input_size=n_feature, hidden_size=tuned_params['hidden_size'],
                                num_layers=tuned_params['n_layers']))
    
    layers.append(Output_lstm())
    layers.append(Reshape_to_linear())
    layers.append(Dropout(tuned_params['dropout']))
    layers.append(Linear(in_features=tuned_params['hidden_size'], out_features=1))

    return torch.nn.Sequential(*layers)

# =============================================================
# Call and train model 
# =============================================================

def train_each_epoch(model, train_loader, loss_function, optimizer, device):
    sum_loss = 0.0
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        pred_outputs = model(inputs)
        targets = targets.reshape((targets.shape[0], 1))
        loss = loss_function(pred_outputs, targets)
        sum_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum_loss / len(train_loader.dataset)

def train_loop(X_train, y_train, hyperparameters, device):

    # extract training and tuned parameters
    batch_size = 32
    num_epochs = hyperparameters['avg_epochs']
    learning_rate = hyperparameters['learning_rate']
    momentum = hyperparameters['weight_decay']
    optimizer_type = hyperparameters['optimizer']

    # transform to tensor 
    tensor_X_train, tensor_y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    tensor_y_train = tensor_y_train.view(len(y_train),1)
    
    # define data loaders for training
    train_loader = DataLoader(dataset=list(zip(tensor_X_train, tensor_y_train)), batch_size=batch_size, shuffle=True) 

    # create and init the model
    model = RNN(hyperparameters)
    model.to(device)

    # define loss function
    loss_function = torch.nn.MSELoss() 

    for epoch in range(num_epochs):
        optimizer = getattr(optim, optimizer_type)(model.parameters(), lr=learning_rate, weight_decay=momentum)
        avg_loss = train_each_epoch(model, train_loader, loss_function, optimizer, device)
        print ('Epoch {}/{}: avg_loss={:.5f}'.format(epoch, num_epochs, avg_loss))
    return model

def run_train_RNN(datapath, X_train, y_train, X_test, y_test, hyperparameters, data_variants, device):

    # preprocessing data
    y_train, y_test = preprocess_mimax_scaler(y_train, y_test)
    
    # training model
    trained_model = train_loop(X_train, y_train, hyperparameters, device)

    # save the trained model
    # print("Saving the trained RNN model at " + "/utils/tuned_RNN.model")
    # torch.save(model, datapath + '/utils/tuned_RNN.model')

    # load the trained model
    # print("Loading the trained RNN model at " + "/utils/tuned_RNN.model")
    # model = torch.load('datapath + '/utils/tuned_RNN.model')

    # ---------------------------------------------
    # Test the trained model with test dataset
    # ---------------------------------------------
    # pca_reloaded = pk.load(open('./pca.pkl', 'rb'))
    # X_test = pca_reloaded.transform(X_test)
    tensor_X_test = torch.Tensor(X_test)
    tensor_X_test = tensor_X_test.to(device)

    trained_model.eval()
    with torch.no_grad():
        
        y_preds = trained_model(tensor_X_test)
        
        if device == torch.device('cpu'):
            y_preds = y_preds.detach().squeeze().numpy()
        else:
            y_preds = y_preds.cpu().detach().squeeze().numpy()

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
    figurename = datapath + '/test_rnn_pheno' + pheno + minmax + standard + pcafitting + '.svg'
    plt.savefig(figurename, bbox_inches='tight')
    print('Saved the figure: ', figurename)

    return trained_model