# Overview
Learn based on turtorial: https://python-bloggers.com/2022/05/building-a-pytorch-binary-classification-multi-layer-perceptron-from-the-ground-up/
<!-- Overview -->   

# Dataset
* Using a dataset from the **MLDataR** project in **Thyroid disease prediction**
* Thyroid data contains:
    - Output is thyroid_class with 0=well and 1=sick

# 1. Creating custom dataset and Building Dataloaders
* Create a class of dataset with one input Dataset. 
* Within the class, build 3 functions:
```python
class (Dataset):    
    def __init__(self,path):
    # Return the number of samples in the dataset
    def __len__(self):
        return len(self.X)
    # Returns a sample from the dataset at the given index idx
    def __getitem__(self,idx):
        return [self.X[idx], self.y[idx]]
```
* Besides, build a custom method to split data into train & test

# 2. Build MLP model
* Base class for all neural network modules
```python
class Model(nn.Module):
    def __init__(self):
        body
    def forward(self, x):
        body
```

# 3. The training loop

```python
def train_model

```

# 4. Evaluation the performance in test dataset

```python
def evaluate_model(test_dl, model):
    body
    return metrics, preds, actuals
```