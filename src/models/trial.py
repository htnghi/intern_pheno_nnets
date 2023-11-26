 for fold, (train_ids, val_ids) in enumerate(kfold.split(X, y)):
        print('FOLD {}'.format(fold))
        
        # transform dataset
        tensor_X_train = torch.Tensor(X[train_ids])
        tensor_X_val = torch.Tensor(X[val_ids])
        tensor_y_train = torch.Tensor(y[train_ids])
        tensor_y_val = torch.Tensor(y[val_ids])

        # unsqueeze 2D array to convert it into 3D array
        tensor_X_train = tensor_X_train.unsqueeze(1)
        tensor_X_val = tensor_X_val.unsqueeze(1)

        # Sample elements randomly from a given list of ids, no replacement.
        # train_subsampler = SubsetRandomSampler(train_ids)
        # val_subsampler = SubsetRandomSampler(val_ids)
        
        # Define data loaders for training and testing data in this fold
        train_loader = DataLoader(dataset=list(zip(tensor_X_train, tensor_y_train)), batch_size=batch_size, shuffle = True)
        val_loader = DataLoader(dataset=list(zip(tensor_X_val, tensor_y_val)), batch_size=batch_size, shuffle = True)

# -------------------------------------------------------------
# 1. Build CNN Model
# -------------------------------------------------------------
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
            Linear(4736, 512),
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
    

# ----------------------------------------------------------
# Sample elements randomly from a given list of ids, no replacement.
train_subsampler = SubsetRandomSampler(train_ids)
val_subsampler = SubsetRandomSampler(val_ids)

# Define data loaders for training and testing data in this fold
train_loader = DataLoader(dataset=list(zip(X, y)), batch_size=batch_size, sampler=train_subsampler)
val_loader = DataLoader(dataset=list(zip(X, y)), batch_size=batch_size, sampler=val_subsampler)