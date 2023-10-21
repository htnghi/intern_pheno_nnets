import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Define class Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

input_image = torch.rand(3,28,28)   # 3 images of size 28x28
# print('Size of input_image: ', input_image.size())

#nn.Flatten
flatten = nn.Flatten()      #nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values
flat_image = flatten(input_image)
print('Size after Flatten: ', flat_image.size())

#nn.Linear
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print('Size after a linear transformation: ', hidden1.size())

#nn.ReLU
hidden1 = nn.ReLU()(hidden1)
print('After ReLU: ', hidden1)

#nn.Sequential
#nn.Sequential is an ordered container of modules. The data is passed through all the modules in the same order as defined
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# nn.Softmax
#  values [0, 1] representing the model’s predicted probabilities for each class.
softmax = nn.Softmax(dim=1)     #dim parameter indicates the dimension along which the values must sum to 1
pred_probab = softmax(logits)


#Tracks all fields defined inside your model object
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():        #access all parameters inside model
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
