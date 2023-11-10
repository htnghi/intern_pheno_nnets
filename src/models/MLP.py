import numpy as np
from numpy import vstack
from pandas import read_csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
from torch.nn import ReLU
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
        self.hidden1 = Linear(n_inputs, 2048)     # a linear transformation (fully connected layer) with n_inputs input features and 20 output features
        #print(self.hidden1.weight[0][5])   #tensor(-0.1823)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')  #initialize the weights
                                                                    #helps prevent the issue of vanishing gradients when using ReLU activations
        #print(self.hidden1.weight.shape)    #torch.Size([20, 26])
        #print(self.hidden1.weight[0][5])   #tensor(-0.2949)
        self.act1 = ReLU()
        #print(self.hidden1.weight[0][5])   #tensor(-0.2949)
        # 2nd hidden layer
        self.hidden2 = Linear(2048, 1024)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # 3rd hidden layer
        self.hidden3 = Linear(1024,1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

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
        return X
    
# ==============================================================
# The trainning loop
# ==============================================================
def train_model(num_epochs, model, train_loader, learning_rate, momentum):
    # Define your optimisation function for reducing loss when weights are calculated 
    # and propogated through the network
    start = time.time()     # keep how long the loop takes
    criterion = MSELoss()   
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    loss = 0.0  # default 0 at the start to initialize the variable

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        #print('Epoch {}/{}'.format(epoch+1, epochs))
        #print('-' * 10)
        model.train()
        # Iterate through training data loader
        for i, (inputs, targets) in enumerate(train_loader):
            #print(targets.shape)
            #print(inputs.shape)
            #print(inputs[5,0], targets[5,0])
            optimizer.zero_grad()   # optimizer sets to 0 gradients
            outputs = model(inputs)
            # Get the prediction value
            loss = criterion(outputs, targets)
            loss.backward()     #set the loss to back propagate through the network updating the weights
            #print('Gradient w.r.t weight after backpropagation:', model.hidden1.weight.grad[0][5])
            #print('weight after backpropagation:', model.hidden1.weight[0][5])
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(f'Gradient for {name}:')
            #         print(param.grad)    
            optimizer.step()
            #print('Gradient w.r.t weight after optimizer', model.hidden1.weight.grad[0][5])
            #print('weight after optimizer', model.hidden1.weight[0][5])
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(f'Gradient for {name}:')
            #         print(param.grad)    

            # Print epoches, batches and losses
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
            .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                
    time_delta = time.time() - start
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_delta // 60, time_delta % 60))
    
    return model

# ==========================================================
# Evaluation the performance of network
# ==========================================================
def evaluate_model(test_loader, model, beta=1.0):
    model.eval()
    with torch.no_grad():
        total_step = len(test_loader)
        for (i, (inputs, targets)) in enumerate(test_loader):
            # Evaluate the model on the test set
            yhat = model(inputs)    # yhat: predictions
            #print('Shape of yhat before detach:', yhat.shape)   

            # Extract the weights using detach to get the numerical values in an ndarray, instead of tensor
            preds = yhat.detach().numpy()  
            #print('Shape of preds before detach:', preds.shape) 

            # Set the actual label to a numpy() array
            actuals = targets.numpy()
            actuals = actuals.reshape((len(actuals), 1))   #Reshape the actual to match what we did at the last part of the data loader class


            print('Test step [{}/{}]: len(test)={}'.format(i + 1, total_step, len(test_loader)))
            print('Preds value = {}, Actual values = {}'.format(preds, actuals))
            print('ExpVar={:7.3f}, R2={:7.3f}, MSE={:7.3f}, MAE={:7.3f}'.format(explained_variance_score(actuals, preds), r2_score(actuals, preds), mean_squared_error(actuals, preds), mean_absolute_error(actuals, preds)))


        # Stack the predictions and actual arrays vertically
        #preds, actuals = vstack(preds), vstack(actuals)     #return a tuple with shape (275,1)
        # print("+ len(preds): ", len(preds))
        # print("+ len(actuals): ", len(actuals))

        # metrics = {
        #     'accuracy': accuracy_score(actuals, preds),
            # 'AU_ROC': roc_auc_score(actuals, preds),
            # 'f1_score': f1_score(actuals, preds),
            # 'average_precision_score': average_precision_score(actuals, preds),
            # 'f_beta': ((1+beta**2) * precision_score(actuals, preds) * recall_score(actuals, preds)) / (beta**2 * precision_score(actuals, preds) + recall_score(actuals, preds)),
            # 'precision': precision_score(actuals, preds),
            # 'recall': recall_score(actuals, preds),
            # 'true_positive_rate_TPR':recall_score(actuals, preds),
        # }

        return preds, actuals


# ==============================================================
# Call and train model
# ==============================================================
def run_train_MLP(X, y):

    # Define relevant hyperparameter for the ML task
    n_inputs = np.size(X, 1) # len of column
    batch_size = 100
    num_epochs = 2
    learning_rate = 0.01
    momentum = 0.9
    
    # transform to torch tensor
    #y1 = y.reshape(len(y), 1)
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    tensor_y = tensor_y.view(500,1)

    # create dataset
    X_train, X_test, y_train, y_test = train_test_split(tensor_x, tensor_y, train_size=0.7, shuffle=True)
    #RNN_dataset = TensorDataset(tensor_x, tensor_y)
    # split train and test
    # train_size = int(0.8 * len(RNN_dataset))
    # test_size = len(RNN_dataset) - train_size
    # train_data, test_data = torch.utils.data.random_split(RNN_dataset, [train_size, test_size])

     # Standardize data
    standard_scaler = StandardScaler()
    # fit scaler
    standard_scaler.fit(X_train)
    # transform training dataset
    X_scaled_train = standard_scaler.transform(X_train)
    print(X_scaled_train.dtype)
    # transform test dataset
    X_scaled_test = standard_scaler.transform(X_test)

    # Dataloader for train and test
    train_loader = DataLoader(dataset = list(zip(X_scaled_train, y_train)), batch_size = batch_size, shuffle = True, num_workers=0)
    test_loader = DataLoader(dataset = list(zip(X_scaled_test, y_test)), batch_size = 50, shuffle = True, num_workers=0)
    # xmatrix,ytarget = next(iter(train_loader))
    # print('Shpe od dataloader:', xmatrix.shape, ytarget.shape)
    # print('Train_data and test_data: ', train_data, test_data)
    # print('Dataloader: ', train_loader)
    exit(1)
    # Call model
    model = MLP(n_inputs)

    # Training model
    train_model(num_epochs, model, train_loader, learning_rate, momentum)

    # Evaluation with test data
    evaluate_model(test_loader, model, beta=1)

    return model

