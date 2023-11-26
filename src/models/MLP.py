import numpy as np
from numpy import vstack
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
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
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU, LeakyReLU, Dropout, BatchNorm1d
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD, Adam
from torch.nn import MSELoss
from torch.optim import lr_scheduler
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import time
import copy
import math


# ==============================================================
# Build MLP Model
# ==============================================================
class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # 1st hidden layer
        self.hidden1 = Linear(n_inputs, 226)     # a linear transformation (fully connected layer) with n_inputs input features and 20 output features
        self.act1 = LeakyReLU()
        # 2nd hidden layer
        self.hidden2 = Linear(226, 193)
        self.act2 = LeakyReLU()
        # 3rd hidden layer
        self.hidden3 = Linear(193, 118)
        self.act3 = LeakyReLU()
        # 4th
        self.hidden4 = Linear(118, 211)
        self.act4 = LeakyReLU()
        # 5th
        self.drop = Dropout(0.4)
        self.hidden5 = Linear(211, 1)
 

    # Forward pass
    def forward(self, X):
        #Input to 1st hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # 2nd hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # 3rd hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # 4th
        X = self.hidden4(X)
        X = self.act4(X)
        # 5th
        X = self.drop(X)
        X = self.hidden5(X)

        return X
    
# ==============================================================
# 2. The trainning loop and function validation
# ==============================================================
def validation(model, val_loader, criterion, val_loss, best_loss, best_weights, val_loss_total, val_exp_variance_sum, val_r2_sum, val_mae_sum):
    
    # define initial values to calculate metrics - val dataset
    val_exp_variance = 0.0
    val_r2 = 0.0
    val_mae = 0.0

    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            yhat = model(inputs)
            loss_validation = criterion(yhat, targets)
            
            # hold the best loss (also the best model)
            if loss_validation < best_loss:
                best_loss = loss_validation
                best_weights = copy.deepcopy(model.state_dict())
            
            # change to numpy for calculating metrics in scikit learn library
            yhat = yhat.detach().numpy() 
            targets = targets.squeeze().numpy()
            # yhat = yhat.flatten().tolist()
            # targets = targets.flatten().tolist()
            # print('r2 score:', r2_score(targets, yhat))

            # collect mse, r2, explained variance from val_dataset
            val_exp_variance = explained_variance_score(targets, yhat)
            val_r2 = r2_score(targets, yhat)
            val_mae = mean_absolute_error(targets, yhat)

            # print metrics after each step in each epoch
            # print ('\t \t  Validation - Step {}: loss = {:.3f}, ExpVar = {:.3f}, R2 = {:.3f}, MAE = {:.3f}'.format(i+1, loss_validation.item(), val_exp_variance, val_r2, val_mae))

            # Sum up mse loss and r2, expVar, mae 
            val_loss += loss_validation.item()
            val_exp_variance_sum += val_exp_variance
            val_r2_sum += val_r2
            val_mae_sum += val_mae

    # Mean validating loss and other metrics for each epoch
    avg_val_loss = val_loss/len(val_loader)
    avg_val_exp_variance = val_exp_variance_sum/len(val_loader)
    avg_val_r2 = val_r2_sum/len(val_loader)
    avg_val_mae = val_mae_sum/len(val_loader)
    print('\t \t Validation - Average: loss = {:.3f}, ExpVar = {:.3f}, R2 = {:.3f}, MAE = {:.3f}'.format(avg_val_loss, avg_val_exp_variance, avg_val_r2, avg_val_mae))

    # storing total loss and metris
    val_loss_total.append(avg_val_loss)

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    # print("\t\t\t Best loss_MSE: %.2f" % best_loss)
    # plt.plot(range(len(train_loss)), train_loss, val_loss)
    # plt.show()

    return val_loss_total



def train_model(num_epochs, X, y, k_folds, batch_size, learning_rate, momentum):

    # number of folds for cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    # declare arrays for storing total loss and other metrics
    train_loss_total = []
    val_loss_total = []

    # define for holding the best model
    best_loss = np.inf
    best_weights = None
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X, y)):
        print('FOLD {}: len(train)={}, len(val)={}'.format(fold, len(train_ids), len(val_ids)))

        # extract X, y for training and validating
        X_train, y_train = X[train_ids], y[train_ids]
        X_val, y_val = X[val_ids], y[val_ids]

        # PCA
        pca = PCA(0.8)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_val = pca.transform(X_val)
        pk.dump(pca, open('./pca.pkl', 'wb'))
        print('shape after PCA: train ={}, val={}'.format(X_train.shape, X_val.shape))

        # Define relevant hyperparameter for the ML task
        n_inputs = np.size(X_train, 1) # len of column

        # transform to tensor 
        tensor_X_train, tensor_y_train = to_tensor(X_train, y_train)
        tensor_X_val, tensor_y_val = to_tensor(X_val, y_val)
        
        # Define data loaders for training and testing data in this fold
        train_loader = DataLoader(dataset=list(zip(tensor_X_train, tensor_y_train)), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=list(zip(tensor_X_val, tensor_y_val)), batch_size=batch_size, shuffle=True)    

        for epoch in range(num_epochs):
            print ('\t Epoch [{}/{}]: Batch size(train)={}, Batch size(val)={}'.format(epoch+1, num_epochs, batch_size, batch_size))

            # Call model
            model = MLP(n_inputs)

            # define loss function and optimizer
            criterion = MSELoss()   
            optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=momentum)

            # define initial values to calculate metrics - test dataset
            train_exp_variance = 0.0
            train_r2 = 0.0
            train_mae = 0.0
            
            # delcare some arrays for storing the sum values of MSE, Exp-variance, R2, MAE
            train_loss = 0.0
            train_exp_variance_sum = 0.0 
            train_r2_sum = 0.0
            train_mae_sum = 0.0

            val_loss = 0.0
            val_exp_variance_sum = 0.0 
            val_r2_sum = 0.0
            val_mae_sum = 0.0

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
                
                # change to numpy for calculating metrics in scikit learn library
                pred_outputs = pred_outputs.detach().numpy() 
                targets = targets.squeeze().numpy()

                # collect mse, r2, explained variance from val_dataset
                train_exp_variance = explained_variance_score(targets, pred_outputs)
                train_r2 = r2_score(targets, pred_outputs)
                train_mae = mean_absolute_error(targets, pred_outputs)

                # print metrics after each step in each epoch
                # print ('\t \t Train - Step {}: loss = {:.3f}, ExpVar = {:.3f}, R2 = {:.3f}, MAE = {:.3f}'.format(i+1, loss_training.item(), train_exp_variance, train_r2, train_mae))

                # Sum up mse loss and r2, expVar, mae 
                train_loss += loss_training.item()
                train_exp_variance_sum += train_exp_variance
                train_r2_sum += train_r2
                train_mae_sum += train_mae
            
            # Mean validating loss and other metrics for each epoch
            avg_train_loss = train_loss/len(train_loader)
            avg_train_exp_variance = train_exp_variance_sum/len(train_loader)
            avg_train_r2 = train_r2_sum/len(train_loader)
            avg_train_mae = train_mae_sum/len(train_loader)
            print('\t \t Training - Average:   loss = {:.3f}, ExpVar = {:.3f}, R2 = {:.3f}, MAE = {:.3f}'.format(avg_train_loss, avg_train_exp_variance, avg_train_r2, avg_train_mae))

            # storing total loss and metris
            train_loss_total.append(avg_train_loss)

            # Validate with val_loader -------------------------------------------------
            validate_each_fold = validation(model, val_loader, criterion, val_loss, best_loss, best_weights, val_loss_total, val_exp_variance_sum, val_r2_sum, val_mae_sum)
        
        print('-----------------------------------------------\n')       

    return model, train_loss_total, val_loss_total


# -------------------------------------------------------------
# 3. Prepare dataset
# -------------------------------------------------------------
# transform dataset to Tensor
def to_tensor(X, y):
    tensor_X = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    return tensor_X, tensor_y

# ==============================================================
# Call and train model
# ==============================================================
def run_train_MLP(datapath, X_train, y_train, X_test, y_test):

    # Define relevant hyperparameter for the ML task
    # n_inputs = np.size(X_train, 1) # len of column
    batch_size = 50
    num_epochs = 5
    k_folds = 5
    learning_rate = 0.000419120031492209
    momentum = 0.007508493751922909
    
    # Get dataset
    # tensor_X, tensor_y = to_tensor(X_train, y_train)
   
    # # Call model
    # model = MLP(n_inputs)

    # Training model
    trained_model = train_model(num_epochs, X_train, y_train, k_folds, batch_size, learning_rate, momentum)

    # Get model
    # specify the zeroth index for the return of model
    model = trained_model[0]

    # save the trained model
    # torch.save(model, datapath + '/utils/save_CNN.model')

     # load the trained model
    # print("Loading the trained CNN model ...\n")
    # model = torch.load('datapath + '/utils/save_CNN.model')

    # Get train loss and val loss for plotting
    # train_loss = trained_model[1]
    # val_loss = trained_model[2]
    # print('Values of training loss: ', train_loss)
    # print('Values of validating loss: ', val_loss)

    # -------------------------------------------------------------
    # Evaluate model by test dataset
    pca_reloaded = pk.load(open('pca.pkl', 'rb'))
    X_test = pca_reloaded.transform(X_test)
    tensor_X_test = torch.Tensor(X_test)

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

        print('\t \t Test - Average:   loss = {:.3f}, ExpVar = {:.3f}, R2 = {:.3f}, MAE = {:.3f}'.format(test_mse, test_exp_variance, test_r2, test_mae))

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

