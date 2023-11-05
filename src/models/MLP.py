from numpy import vstack
from pandas import read_csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
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
        self.hidden1 = Linear(n_inputs, 20)     # a linear transformation (fully connected layer) with n_inputs input features and 20 output features
        #print(self.hidden1.weight[0][5])   #tensor(-0.1823)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')  #initialize the weights
                                                                    #helps prevent the issue of vanishing gradients when using ReLU activations
        #print(self.hidden1.weight.shape)    #torch.Size([20, 26])
        #print(self.hidden1.weight[0][5])   #tensor(-0.2949)
        self.act1 = ReLU()
        #print(self.hidden1.weight[0][5])   #tensor(-0.2949)
        # 2nd hidden layer
        self.hidden2 = Linear(20, 10)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # 3rd hidden layer
        self.hidden3 = Linear(10,4)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    # Forward pass
    def forward(self, X):
        #Input to 1st hidden layer
        X = self.hidden1(X)
        #print('Hidden1: ', self.hidden1.weight[0][5])
        X = self.act1(X)
        #print('After ReLU: ', self.hidden1.weight[0][5])
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
def train_model(train_loader, model, epochs, lr, momentum):
    # Define your optimisation function for reducing loss when weights are calculated 
    # and propogated through the network
    start = time.time()     # keep how long the loop takes
    criterion = BCELoss()   # calculate the Loss by binary cross entropy(BCELoss)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    loss = 0.0  # default 0 at the start to initialize the variable

    for epoch in range(epochs):
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
            # Get the class labels(preds)
            _, preds = torch.max(outputs.data,1)   # torch.max return 2 values (max_values, index). #dim=1 => maximum in each row
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
    time_delta = time.time() - start
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_delta // 60, time_delta % 60))
    
    return model


# ==============================================================
# Call and train model
# ==============================================================
def run_train_MLP(X, y):

    # Define relevant hyperparameter for the ML task
    n_inputs = 2000
    batch_size = 100
    num_epochs = 2
    learning_rate = 0.01
    momentum = 0.9
    
    # transform to torch tensor
    #y1 = y.reshape(len(y), 1)
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y)

    # create dataset
    RNN_dataset = TensorDataset(tensor_x, tensor_y)
    # split train and test
    train_size = int(0.8 * len(RNN_dataset))
    test_size = len(RNN_dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(RNN_dataset, [train_size, test_size])

    # Dataloader for train and test
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, num_workers=0)
    test_loader = DataLoader(dataset = test_data, batch_size = 10000, shuffle = True, num_workers=0)
    # xmatrix,ytarget = next(iter(train_loader))
    # print('Shpe od dataloader:', xmatrix.shape, ytarget.shape)
    # print('Train_data and test_data: ', train_data, test_data)
    # print('Dataloader: ', train_loader)

    # Call model
    model = MLP(n_inputs).to(device)

    # Training model
    train_model(num_epochs, model, train_loader, learning_rate, momentum)

    return model