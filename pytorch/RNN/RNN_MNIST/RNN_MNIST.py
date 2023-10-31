import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
from numpy import vstack
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score
from sklearn.metrics import confusion_matrix, recall_score, f1_score
import math


# Device will determine whether to run the training on GPU or CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==============================================================
# Build RNN Model
# ==============================================================
class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Passing in the input and hidden state into the model and  obtaining outputs
        out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out       
    
# ==============================================================
# The trainning loop
# ==============================================================

def train(num_epochs, model, train_loader):
    
    # Define loss function
    loss_func = nn.CrossEntropyLoss()

    # Define optimization function
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)    

    # Train the model
    total_step = len(train_loader)
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            
            # Reshape images
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_func(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad() # optimizer sets to 0 gradients so that you do the parameter update correctly
            loss.backward()
            optimizer.step()
            
            # Print epoches, batches and losses
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))   

# ==============================================================
# Evaluating the model
# ==============================================================
def evaluate_model(test_loader, model):
    model.eval()
    with torch.no_grad():
        preds = []
        actuals = []
        for images, labels in test_loader:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)

            # Get the class labels(preds)
            # torch.max return 2 values (max_values, index). #dim=1 => maximum in each row
            _, yhat = torch.max(outputs.data, 1)
            # Using detach to get the numerical values in an ndarray, instead of tensor
            yhat = yhat.detach().numpy()
            # total = total + labels.size(0)
            # correct = correct + (preds == labels).sum().item()
            # print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

            # Set the actual label to a numpy() array
            actual = labels.numpy()
            # actual = actual.reshape((len(actual), 1))
            preds.extend(yhat)
            actuals.extend(actual)

            # Stack the predictions and actual arrays vertically
        
    preds, actuals = vstack(preds), vstack(actuals)
    print("+ len(preds): ", len(preds))
    print("+ len(actuals): ", len(actuals))
    
    # Calculate metrics
    print(confusion_matrix(actuals, preds))
    exit(1)
    
    # Get descriptions of tn, fp, fn, tp
    tn, fp, fn, tp = cm.ravel()
    total = sum(cm.ravel())

    metrics = {
        'accuracy': accuracy_score(actuals, preds, average='weighted'),
        'f1_score': f1_score(actuals, preds),
        'average_precision_score': average_precision_score(actuals, preds),
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



# ==============================================================
# Set hyperparameters
# ==============================================================
# Define relevant variables for the ML task
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# ==============================================================
# Load dataset
# ==============================================================

# Create Training and Testing dataset
train_data = datasets.MNIST(root = './data', train = True, transform = ToTensor(), download = True)
test_data = datasets.MNIST(root = './data', train = False, transform = ToTensor(), download = True)

# Dataloader for train and test
train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, num_workers=0)
test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = True, num_workers=0)

# Visualization of MNIST dataset
# Plot one train_data
# plt.imshow(train_data.data[0], cmap='gray')
# plt.title('%i' % train_data.targets[0])
# plt.show()
# Plot multiple train data
# figure = plt.figure(figsize=(10, 8))
# cols, rows = 5, 5
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(train_data), size=(1,)).item()
#     img, label = train_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(label)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

# ==============================================================
# Call and train model
# ==============================================================
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
print(model)

# train(num_epochs, model, train_loader)

# save the trained model
# torch.save(model, "./trained_model/pytorch_rnn_mnist.model")

# load the trained model
model = torch.load("./trained_model/pytorch_rnn_mnist.model")

# Test the model
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, sequence_length, input_size).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total = total + labels.size(0)
#         correct = correct + (predicted == labels).sum().item()
# print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


# get some random training images
results = evaluate_model(test_loader, model)
# sample = next(iter(test_loader))
# imgs, lbls = sample

# test_output = model(imgs[:10].view(-1, 28, 28))
# predicted = torch.max(test_output, 1)[1].data.numpy().squeeze()
# labels = lbls[:10].numpy()
# print(f"Predicted number: {predicted}")
# print(f"Actual number: {labels}")