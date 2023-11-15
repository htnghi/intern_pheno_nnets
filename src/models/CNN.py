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
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU, Softplus
from torch.nn import Sigmoid
from torch.nn import Module, Sequential
from torch.nn import MaxPool1d, Conv1d, Flatten, LeakyReLU
from torch.optim import SGD, Adam
from torch.nn import MSELoss
from torch.optim import lr_scheduler
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import time
import copy
import math


# ==============================================================
# 1. Build MLP Model
# ==============================================================
class CNN1D(Module):
    def __init__(self, n_outputs):

        super(CNN1D, self).__init__()

        self.model = Sequential(
            Conv1d(1, 16,   kernel_size=3, stride=1, padding=1),
            MaxPool1d(2),

            Conv1d(16, 64,  kernel_size=3, stride=1, padding=1),
            MaxPool1d(2),

            Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            MaxPool1d(2),
        )

        self.linear = Sequential(
            Linear(160000, 512),
            LeakyReLU(inplace=True),
            Linear(512, n_outputs),
        )


    def forward(self,X):
        X = self.model(X)
        # print('shape after conv:', X.shape) #torch.Size([100, 128, 1250])

        X = X.view(X.size(0), -1)
        # print('shape after flatten:', X.shape)  #torch.Size([100, 160000])

        X = self.linear(X)

        return X
    
# ==============================================================
# 2. The trainning loop (including validation)
# ==============================================================
def train_model(num_epochs, model, train_loader, val_loader, learning_rate):
    # define your optimisation function for reducing loss when weights are calculated
    start = time.time()

    # define loss function and optimizer
    criterion = MSELoss() 
    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_loss = 0.0
    val_loss = 0.0
    train_loss_per_epoch = []
    val_loss_per_epoch = []

    # define for holding the best model
    history = []
    best_loss = np.inf
    best_weights = None

    for epoch in range(num_epochs):
        
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
            # print epoches, batches and losses
            print ('Epoch [{}/{}], step {}: training loss = {:.4f}'.format(epoch+1, num_epochs, i+1, loss_training.item()))

            # Mean training loss of each epoch
            train_loss += loss_training.item() 
        
        # Get training loss for each epoch
        avg_train_loss = train_loss/len(train_loader)
        #train_loss_per_epoch.append(avg_train_loss)
        print('Epoch {}: Average training loss = {:.4f}'.format(epoch+1, avg_train_loss))

        #print('Training loss for all epoch', train_loss_per_epoch)
 
        # Validate at end of each epoch ----------------------------------------------
        model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                preds = model(inputs)
                loss_validation = criterion(preds, targets)
                
                # hold the best loss (also the best model)
                if loss_validation < best_loss:
                    best_loss = loss_validation
                    best_weights = copy.deepcopy(model.state_dict())

                # print epoches, batches and losses
                print ('Epoch [{}/{}], step {}: validating loss = {:.4f}'.format(epoch+1, num_epochs, i+1, loss_validation.item()))

                # Mean validating loss of each epoch
                val_loss += loss_validation.item()

                
        # Get validating loss for each epoch
        avg_val_loss = val_loss/len(val_loader)
        val_loss_per_epoch.append(avg_val_loss)
        print('Epoch {}: Average validating loss = {:.4f}'.format(epoch+1, avg_val_loss))
        #print('Validating loss for all epoch', val_loss_per_epoch)

        # restore model and return best accuracy
        model.load_state_dict(best_weights)
        print("Best loss_MSE: %.2f" % best_loss)
        # plt.plot(history)
        # plt.plot(range(len(train_loss)), train_loss, val_loss)
        # plt.show()

    print('-----------------------------------------------\n')       
    training_time = time.time() - start
    # print('Training complete in {:.0f}m {:.0f}s'.format(training_time // 60, training_time % 60))
    
    return model, train_loss_per_epoch, val_loss_per_epoch

# ************************************************************************************************************************************
# ==============================================================
# Call and train model
# ==============================================================
def run_train_CNN(X, y):

    # define relevant hyperparameter for the ML task
    n_outputs  = 1
    batch_size = 100
    num_epochs = 2
    learning_rate = 0.0005

    # ----------------------------------------------------------
    # Standardize data
    # ----------------------------------------------------------
    standard_scaler = StandardScaler()
    standard_scaler.fit(X)
    X_scaled = standard_scaler.transform(X)

    minmax_scaler = MinMaxScaler()
    y = np.expand_dims(y, axis=1)
    y_scaled = minmax_scaler.fit_transform(y)

    # print("Data after normalization: ")
    # print(X_scaled)
    # print(y_scaled)
    # print('-----------------------------------------------\n')

    # ----------------------------------------------------------
    # Try to do PCA here
    # ----------------------------------------------------------
    
    # split dataset
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.8, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, train_size=0.8, shuffle=True)

    # transform training dataset
    tensor_X_train = torch.Tensor(X_train)
    # transform test dataset
    tensor_X_val = torch.Tensor(X_val)
    
    # unsqueeze 2D array to convert it into 3D array
    tensor_X_train = tensor_X_train.unsqueeze(1)
    tensor_X_val  = tensor_X_val.unsqueeze(1)
    tensor_y_train = torch.Tensor(y_train) # .view(len(y_train),1)
    tensor_y_val  = torch.Tensor(y_val)  # .view(len(y_val),1)

    # Dataloader for train and test
    train_loader = DataLoader(dataset=list(zip(tensor_X_train, tensor_y_train)), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=list(zip(tensor_X_val, tensor_y_val)), batch_size=45, shuffle=True, num_workers=0)

    # Call model
    model = CNN1D(n_outputs)

    # Training model
    trained_model = train_model(num_epochs, model, train_loader, val_loader, learning_rate)

    # Get model
    # specify the zeroth index for the return of model
    model = trained_model[0]

    return model