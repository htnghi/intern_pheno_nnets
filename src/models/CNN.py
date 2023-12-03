import numpy as np
from numpy import vstack
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
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
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU, Softplus
from torch.nn import Sigmoid
from torch.nn import Module, Sequential
from torch.nn import MaxPool1d, Conv1d, Flatten, LeakyReLU, BatchNorm1d, Dropout
from torch.optim import SGD, Adam
from torch.nn import MSELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import time
import copy
import math


# -------------------------------------------------------------
# 1. Build CNN Model
# -------------------------------------------------------------
class CNN1D(Module):
    def __init__(self, n_inputs):

        super(CNN1D, self).__init__()

        self.model = Sequential(
            Conv1d(n_inputs, 4,   kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm1d(4),

            Conv1d(4, 16,  kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm1d(16),

            Conv1d(16, 64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm1d(64),
            MaxPool1d(2),
        )

        self.linear = Sequential(
            Linear(320000, 512),
            LeakyReLU(inplace=True),
            BatchNorm1d(512),
            Dropout(0.5),
            Linear(512, 1),
        )


    def forward(self,X):
        # print('shape initial:', X.shape) #torch.Size([batch_size=80, 10000, 4])
        
        
        X = self.model(X)
        # print('shape after conv:', X.shape) #torch.Size([batch_size=50, 64, 150])

        X = X.view(X.size(0), -1)
        # print('shape after flatten:', X.shape)  #torch.Size([batch_size, 9600])

        X = self.linear(X)

        return X

# -------------------------------------------------------------
# 2. The trainning loop and function validation
# -------------------------------------------------------------
def train_each_epoch(train_loader, model, criterion, optimizer, train_loss, train_loss_total):

    # training loop with train_loader --------------------------------------------
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

        # Sum up mse loss and r2, expVar, mae 
        train_loss += loss_training.item()
    
    # Mean validating loss and other metrics for each epoch
    train_loss_avg = train_loss/len(train_loader)
    print('\t \t Training - Average: loss = {:.3f}'.format(train_loss_avg))

    # storing total loss and metris
    train_loss_total.append(train_loss_avg)

    return model, train_loss_total


def validation_each_epoch(model, val_loader, criterion, val_loss, val_loss_total):

    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            yhat = model(inputs)
            loss_validation = criterion(yhat, targets)
            
            # Sum up loss
            val_loss += loss_validation.item()

    # Mean validating loss for each epoch
    val_loss_avg = val_loss/len(val_loader)
    print('\t \t Validation - Average: loss = {:.3f}'.format(val_loss_avg))

    # storing total loss and metris
    val_loss_total.append(val_loss_avg)

    return val_loss_avg, val_loss_total


def train_model(num_epochs, X, y, k_folds, batch_size, learning_rate, momentum):

    # number of folds for cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    # declare arrays for storing total loss and other metrics
    train_loss_total = []
    val_loss_total = []

    # define for holding the best model
    best_loss = np.inf
    best_weights = None
    best_epoch = 0
    best_fold  = 0
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X, y)):
        print('FOLD {}: len(train)={}, len(val)={}'.format(fold+1, len(train_ids), len(val_ids)))

        # extract X, y for training and validating
        X_train, y_train = X[train_ids], y[train_ids]
        X_val, y_val = X[val_ids], y[val_ids]

        # MinMax Scaler
        minmax_scaler = MinMaxScaler()
        y_train = np.expand_dims(y_train, axis=1)
        y_train_scaled = minmax_scaler.fit_transform(y_train)
        y_val = np.expand_dims(y_val, axis=1)
        y_val_scaled = minmax_scaler.fit_transform(y_val)

        # PCA
        # pca = PCA(n_components=218)
        # pca.fit(X_train)
        # X_train = pca.transform(X_train)
        # X_val = pca.transform(X_val)
        # pk.dump(pca, open('./pca.pkl', 'wb'))
        # print('shape after PCA: train ={}, val={}'.format(X_train.shape, X_val.shape))

        # Define relevant hyperparameter for the CNN task
        # n_inputs = 1 # for additive encoding
        n_inputs = 4 # for onehot encoding

        # transform to tensor 
        tensor_X_train, tensor_y_train = to_tensor(X_train, y_train_scaled)
        tensor_X_val, tensor_y_val = to_tensor(X_val, y_val_scaled)
        
        # tensor_X_train, tensor_X_val = tensor_X_train.unsqueeze(1), tensor_X_val.unsqueeze(1) # for additive encoding
        tensor_X_train, tensor_X_val = torch.swapaxes(tensor_X_train, 1, 2), torch.swapaxes(tensor_X_val, 1, 2)

        # Define data loaders for training and testing data in this fold
        train_loader = DataLoader(dataset=list(zip(tensor_X_train, tensor_y_train)), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=list(zip(tensor_X_val, tensor_y_val)), batch_size=batch_size, shuffle=True)  

        # Call model
        model = CNN1D(n_inputs)

        # define loss function and optimizer
        criterion = MSELoss()   
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=momentum)  

        for epoch in range(num_epochs):
            print ('\t Epoch [{}/{}]: Batch size(train)={}, Batch size(val)={}'.format(epoch+1, num_epochs, batch_size, batch_size))

            # delcare some arrays for storing the sum values of MSE
            train_loss = 0.0
            val_loss = 0.0

            # training loop with train_loader --------------------------------------------
            model, train_loss_total = train_each_epoch(train_loader, model, criterion, optimizer, train_loss, train_loss_total)

            # Validate with val_loader -------------------------------------------------
            val_loss_avg, val_loss_total = validation_each_epoch(model, val_loader, criterion, val_loss, val_loss_total)
        
            # hold the best loss (also the best model)
            if val_loss_avg < best_loss:
                best_fold = fold+1
                best_epoch = epoch+1
                best_loss = val_loss_avg
                best_weights = copy.deepcopy(model.state_dict())

        # restore model
        model.load_state_dict(best_weights)
    print('Best loss_MSE={:.4f} at fold {} and epoch {}'.format(best_loss, best_fold, best_epoch))
    print('-----------------------------------------------\n')       

    return model, train_loss_total, val_loss_total

# -------------------------------------------------------------
# 3. Functions for preprocessing
# -------------------------------------------------------------
# Min Max Scaler
def minmax_scaler(y):
    minmax_scaler = MinMaxScaler()
    y = np.expand_dims(y, axis=1)
    y_scaled = minmax_scaler.fit_transform(y)
    return  y_scaled

# transform dataset to Tensor
def to_tensor(X, y):
    tensor_X = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    return tensor_X, tensor_y

# ==============================================================
# Call and train model
# ==============================================================
def run_train_CNN(datapath, X_train, y_train, X_test, y_test):

    # Define relevant hyperparameter for the ML task
    # n_inputs = np.size(X_train, 1) # len of column
    
    batch_size = 80
    num_epochs = 2
    k_folds = 2
    learning_rate = 0.0051522889065098165
    momentum = 0.009223144758744747
    
    # Get dataset
    # tensor_X, tensor_y = to_tensor(X_train, y_train)
   
    # # Call model
    # model = CNN1D(n_inputs)

    # Training model
    trained_model = train_model(num_epochs, X_train, y_train, k_folds, batch_size, learning_rate, momentum)

    # Get model
    # specify the zeroth index for the return of model
    model = trained_model[0]

    # save the trained model
    # torch.save(model, datapath + '/utils/save_MLP.model')

     # load the trained model
    # print("Loading the trained CNN model ...\n")
    # model = torch.load('datapath + '/utils/save_MLP.model')

    # Get train loss and val loss for plotting
    # train_loss = trained_model[1]
    # val_loss = trained_model[2]
    # print('Values of training loss: ', train_loss)
    # print('Values of validating loss: ', val_loss)

    # -------------------------------------------------------------
    # Evaluate model by test dataset
    # pca_reloaded = pk.load(open('pca.pkl', 'rb'))
    # X_test = pca_reloaded.transform(X_test)
    tensor_X_test = torch.Tensor(X_test)

    # Encoding
    # tensor_X_test = tensor_X_test.unsqueeze(1) # only for additive encoding
    tensor_X_test = torch.swapaxes(tensor_X_test, 1, 2)

    y_test = minmax_scaler(y_test)

    model.eval()
    with torch.no_grad():
        y_preds = model(tensor_X_test)

        # change to numpy for calculating metrics in scikit learn library
        y_preds = y_preds.detach().squeeze().numpy()
        y_test  = y_test.squeeze()

        # collect mse, r2, explained variance
        test_mse = mean_squared_error(y_test, y_preds)
        test_exp_variance = explained_variance_score(y_test, y_preds)
        test_r2 = r2_score(y_test, y_preds)
        test_mae = mean_absolute_error(y_test, y_preds)

        print('\t \t Test - Average:   loss = {:.4f}, ExpVar = {:.4f}, R2 = {:.4f}, MAE = {:.4f}'.format(test_mse, test_exp_variance, test_r2, test_mae))

    # Plot the model
    x_plot = np.arange(len(y_preds))
    plt.scatter(x_plot, y_test, alpha=0.5, label='ground_true')
    plt.plot(x_plot, y_preds, label='prediction', color='r')

    plt.xlabel('Samples', fontsize=13)
    plt.ylabel('Pheno1 (g)', fontsize=13)
    plt.grid()
    plt.legend()
    plt.savefig(datapath + '/utils/MLP_preds_groundtrue.svg', bbox_inches='tight')
    plt.show()
 

    return model


 





