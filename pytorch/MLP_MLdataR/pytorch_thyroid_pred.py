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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataloaders
# ================================================
class ThyroidCSVDataset(Dataset):
    def __init__(self,path):
        df = read_csv(path, header=None)
        # Store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]   #outcome variable is in the last column
        self.X = self.X.astype('float32')
        # Label encode the target as values 1:sick and 0:not sick
        self.y = LabelEncoder().fit_transform(self.y)   #from sklearn library
        self.y = self.y.astype('float32')       # transform into Float tensor
        #print('Shape y before reshape:', self.y.shape)   #(2750,)
        self.y = self.y.reshape((len(self.y), 1))   #converts a 1D array into a 2D array
                                                    # compatible with the expected format of the target variable when used with PyTorch's data loading utilities
        #print('Shape y after reshape:', self.y.shape)  #(2750,1)
    # Return the number of samples in the dataset
    def __len__(self):
        #print('Shape of dataset X: ', self.X.shape)    #(2750,26)
        #print('Number of samples in dataset: ', len(self.X))   #2750
        #print('Datatype of dataset: ', self.X.dtype)
        return len(self.X)
   
    
    # Returns a sample from the dataset at the given index idx
    def __getitem__(self,idx):
        #?????????????????????
        return [self.X[idx], self.y[idx]]
    

    # Create train data and test data
    def split_data(self, split_ratio=0.2):
        test_size = round(split_ratio * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])
    
# Build MLP model
# ===========================================================
# Create model
class ThyroidMLP(Module):
    def __init__(self, n_inputs):
        super(ThyroidMLP, self).__init__()
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
        self.hidden3 = Linear(10,1)
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
    
# The training loop
# ===============================================================
# Create training loop
def train_model(train_dl, model, epochs=100, lr=0.01, momentum=0.9, save_path='thyroid_best_model.pth'):
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
        for i, (inputs, targets) in enumerate(train_dl):
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
        torch.save(model, save_path)
    time_delta = time.time() - start
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_delta // 60, time_delta % 60))
    
    return model

# Evaluation the performance of network
# ==========================================================
def evaluate_model(test_dl, model, beta=1.0):
    preds = []
    actuals = []

    for (i, (inputs, targets)) in enumerate(test_dl):
        # Evaluate the model on the test set
        yhat = model(inputs)    # yhat: predictions
        #print('Shape of yhat before detach:', yhat.shape)   #torch.Size([275, 1])

        # Extract the weights using detach to get the numerical values in an ndarray, instead of tensor
        yhat = yhat.detach().numpy()  
        #print('Shape of yhat before detach:', yhat.shape)   #(275,1)

        # Set the actual label to a numpy() array
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))   #Reshape the actual to match what we did at the last part of the data loader class
       
        # Round to get the class value i.e. sick vs not sick
        #print('Value yhat before round:', yhat[5,:])   #[0.6910267]
        yhat = yhat.round()
        #print(Value yhat after round:', yhat[5,:]) #[1.]

        # Store the predictions in the empty lists initialised at the start of the class
        preds.append(yhat)  #get a list of preds
        actuals.append(actual)

    # Stack the predictions and actual arrays vertically
    preds, actuals = vstack(preds), vstack(actuals)     #return a tuple with shape (275,1)
    #Calculate metrics
    cm = confusion_matrix(actuals, preds)
    # Get descriptions of tp, tn, fp, fn
    tn, fp, fn, tp = cm.ravel()
    total = sum(cm.ravel())

    metrics = {
        'accuracy': accuracy_score(actuals, preds),
        'AU_ROC': roc_auc_score(actuals, preds),
        'f1_score': f1_score(actuals, preds),
        'average_precision_score': average_precision_score(actuals, preds),
        'f_beta': ((1+beta**2) * precision_score(actuals, preds) * recall_score(actuals, preds)) / (beta**2 * precision_score(actuals, preds) + recall_score(actuals, preds)),
        'matthews_correlation_coefficient': (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),
        'precision': precision_score(actuals, preds),
        'recall': recall_score(actuals, preds),
        'true_positive_rate_TPR':recall_score(actuals, preds),
        'false_positive_rate_FPR':fp / (fp + tn) ,
        'false_discovery_rate': fp / (fp +tp),
        'false_negative_rate': fn / (fn + tp) ,
        'negative_predictive_value': tn / (tn+fn),
        'misclassification_error_rate': (fp+fn)/total ,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        #'confusion_matrix': confusion_matrix(actuals, preds), 
        'TP': tp,
        'FP': fp, 
        'FN': fn, 
        'TN': tn
    }
    return metrics, preds, actuals
        

# Create the prediction function
# ================================================
def predict(row, model):
    row = Tensor([row])
    yhat = model(row)
    # Get numpy array
    yhat = yhat.detach().numpy()
    return yhat   

# Prepare data to use with the model
# ================================================
def prepare_thyroid_dataset(path):
    dataset = ThyroidCSVDataset(path)
    train, test = dataset.split_data(split_ratio=0.1)
    # Prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    #print(len(train)) #2475
    #print(len(train_dl))    #78
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


# Load & Use Model
# ================================================

# Load imbalanced dataset
# **************************
train_dl, test_dl = prepare_thyroid_dataset('https://raw.githubusercontent.com/StatsGary/Data/main/thyroid_raw.csv')

# Train model
model = ThyroidMLP(26)

train_model(train_dl, 
            model, 
            save_path='data/thyroid_model.pth', 
            epochs=150, 
            lr=0.01)


# Evaluation with test data
results = evaluate_model(test_dl, model, beta=1)
model_metrics = results[0]  #specify the zeroth index of the return to get metrics
metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index', columns=['metric'])  #get dictionary type for metrics
metrics_df.index.name = 'metric_type'   #give name for colume of metric names
metrics_df.reset_index(inplace=True)    #give indexes
metrics_df.to_csv('confusion_matrix_thyroid.csv', index=False)

# Prediction a new dataset
row = [0.8408678952719717,0.7480132415430958,-0.3366221139379705,-0.0938130059640389,-0.1101874782051067,-0.2098160394213988,-0.1260114177378201,-0.1118651062104989,-0.1274917875477927,-0.240146053214037,-0.2574472174396955,-0.0715198539852151,-0.0855764265990022,-0.1493202733578882,-0.0190692517849118,-0.2590488060984638,0.0,-0.1753175780014474,0.0,-0.9782211033008232,0.0,-1.3237957945784953,0.0,-0.6384998731458282,0.0,-1.209042232192488]
yhat = predict(row, model)
print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))

# Load balanced dataset
# **************************

# Get the ionsphere data
train_dl, test_dl = prepare_thyroid_dataset('https://raw.githubusercontent.com/StatsGary/Data/main/ion.csv')

# Train the model
# Specify the number of input dimensions
model = ThyroidMLP(34)

train_model(train_dl, model, 
            save_path='data/ionsphere_model.pth', 
            epochs=150, 
            lr=0.01)

# Evaluation with test data
def eval_model(test_dl, model, cm_out_name='confusion_mat.csv',
               beta=1, export_index=False):
    results = evaluate_model(test_dl, model, beta)
    model_metrics = results[0]
    metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index', columns=['metric'])
    metrics_df.index.name = 'metric_type'
    metrics_df.reset_index(inplace=True)
    metrics_df.to_csv(cm_out_name, index=export_index)
    #print(metrics_df)
    return metrics_df, model_metrics, results

results = eval_model(test_dl, model)
#print(results[0])

# Prediction a new dataset
row = [1,0,1,-0.18829,0.93035,-0.36156,-0.10868,-0.93597,1,-0.04549,0.50874,-0.67743,0.34432,-0.69707,-0.51685,-0.97515,0.05499,-0.62237,0.33109,-1,-0.13151,-0.45300,-0.18056,-0.35734,-0.20332,-0.26569,-0.20468,-0.18401,-0.19040,-0.11593,-0.16626,-0.06288,-0.13738,-0.02447]
yhat = predict(row, model)
print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))

