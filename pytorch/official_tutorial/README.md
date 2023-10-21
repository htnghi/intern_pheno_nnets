# 1. Tensors
* Tensor is a multi-dimensional matrix containing elements of a single data type. 
* Tensors are similar to NumPy arrays, but they have additional capabilities, especially when it comes to GPU acceleration and automatic differentiation, which is crucial for training deep learning models.
* Initializing a Tensor
    - Directly from data
    - From Numpy array
    - From another tensor: the new tensor retains the properties (shape, datatype) of the argument tensor
    - With random or constant values: e.g. random, 1, 0.
* Attribute: describe their **shape, datatype, and the device** on which they are stored.

# 2. Datasets & Dataloaders
* **Dataset** stores the samples and their corresponding labels.
    - a part of the `torch.utils.data` module.
    - Key features and benefits:
        + Standardized a custom dataset: must implement three functions: __init__, __len__, and __getitem__.
        + 
* **DataLoader** wraps an iterable around the Dataset to enable easy access to the samples.
    - a utility that provides efficient and convenient ways to load and iterate over large datasets. 
    - a part of the `torch.utils.data` module.
    - Key features and benefits:
        + Batching: DataLoaders allow you to split a dataset into batches of data => stochastic gradient descent and mini-batch training.
        + Shuffling: DataLoaders can shuffle the data within each epoch, ensuring that the order of the training samples is different in each iteration => avoiding patterns and biases that may result from data order.
        + Parallel Data Loading: DataLoaders can automatically load data in parallel => speed up data loading.
        + Efficient Memory Management: DataLoaders help manage memory efficiently. Instead of loading the entire dataset into memory at once, they load and process data in small batches => reducing memory requirements.

# 3. Transform
* transforms are used to perform some manipulation of the data and make it suitable for training.

# 4. Build Model
* The torch.nn namespace provides all the building blocks that need to build a neural network.
* Define neural network by subclassing nn.Module

# 5. Autograd

# 6. Optimization
* **Hyperparameters** are adjustable parameters that let you control the model optimization process.\
The following hyperparameters for training:
    - Number of Epochs - the number times to iterate over the dataset
    - Batch Size - the number of data samples propagated through the network before the parameters are updated
    - Learning Rate - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.
* Optimization loop:
    - we can then train and optimize our model with an optimization loop. 
    - Each iteration of the optimization loop is called an **epoch**. Each epoch consists: the train loop and the Validation/ Test loop.
* Optimization is the process of adjusting model parameters to reduce model error in each training step.\
    - Optimization algorithms define how this process is performed, i.e. GSD, ADAM, RMSProp
* Note: All optimization logic is encapsulated in the `optimizer` object.
˜ı
# 7. Save and Load the model


# Some useful code 
* Access the gradients of weights and biases for each layer
```python
for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f'Gradient for {name}:')
                    print(param.grad)
```








