import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

#============================================================================#
#                          Tensor                                            #
#============================================================================#

# 1. Initializing a Tensor
## Directly from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
#print('Tensor from data: ', x_data)

## From Numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
#print('Tensor from Numpy array: ', x_np)

## From another tensor
x_ones = torch.ones_like(x_data)    #retains the properties of x_data
#print('Ones Tensor: ', x_ones)
x_rand = torch.rand_like(x_data, dtype=torch.float)     #overrides the datatype of x_data
#print('Random Tensor: ', x_rand)

## With random or constant values
shape = (2,3,)  #tensor dimensionality
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

#print('Random Tensor: ', rand_tensor)
#print('Ones Tensor: ', ones_tensor)
#print('Zeros Tensor: ', zeros_tensor)

# 2. Atrributes
tensor = torch.rand(3,4)

#print('Shape of tensor: ', tensor.shape)
#print('Datatype of tensor: ', tensor.dtype)
#print('Device tensor is stored on: ', tensor.device)

#3. Operations:
## Indexing and Slicing:
tensor = torch.ones(4,4)
#print('1st row: ', tensor[0])
#print('1st column: ', tensor[:, 0])
#print('last column: ', tensor[..., -1])
tensor[:,1] = 0
#print(tensor)

## Joining tensor
t1 = torch.cat([tensor, tensor, tensor], dim=1)
#print('Joining tensor: ', t1)

## Arithmetic operations: torch.matmul and torch.mul
y1 = tensor @ tensor.T 
y2 = tensor.matmul(tensor.T)    #matrix multiplication
#print('y1: ', y1)
#print('y2: ', y2)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
#print('y3: ', y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)     # multiple element-wise product
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

## Single-element tensors (can convert it to Python numerical value)
agg = tensor.sum()
agg_item = agg.item()
#print(agg_item, type(agg_item))

## In-place operations (denoted by a _ suffix)
tensor.add_(5)
#print(tensor)

# 3. Bridge with Numpy
## Tensor to Numpy
t = torch.ones(5)
#print('tensor: ', t)
n = t.numpy()
#print('Numpy (from torch): ', n)

t.add_(1)   # change in tensor: reflects in numpy array
#print('Numpy reflected: ', n)

## Numpy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)     #change in numpy array: reflects in tensor
#print('Tensor reflected: ', t)

