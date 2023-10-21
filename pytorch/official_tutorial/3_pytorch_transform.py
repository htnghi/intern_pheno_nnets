import torch
from orchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),       # parameter 'transform': modify the features      
                                # ToTensor converts a PIL image or NumPy ndarray into a FloatTensor
                                # scales the image’s pixel intensity values in the range [0., 1.]
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
                                # parameter 'target_transform': modify labels
                                # Lambda transforms apply any user-defined lambda function.
)