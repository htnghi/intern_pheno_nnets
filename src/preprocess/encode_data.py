import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import one_hot

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Read the prepared (raw) data by pandas
def read_data(datapath):
    """
    Read the raw data or impured data by pandas,
        + in: temporarily the data path is fixed
        + out: the returned format is pandas dataframe and write to .csv files
    """

    df_genotypes  = pd.read_csv(datapath + '/data/add012encoded_geneotype_dataset.csv')
    df_phenotypes = pd.read_csv(datapath + '/data/phenotype_data.csv')

    # Some basis summary about the dataset
    # print('------------------------------------------------------------------')
    # print('Number of Samples:\t%d' %df_genotypes.shape[0]) #2029
    # print('Number of snp:\t%d' %(df_genotypes.shape[1]-1)) #10001
    # print('Number of Corresponding phenotype values:\t%d' %df_phenotypes.shape[0]) #2030
    # print('------------------------------------------------------------------\n')

    # delete nan values from phenotye data
    df_pheno = df_phenotypes.dropna(subset=['sample_ids', 'pheno1']) #only delte missing value in pheno1 column
    # print('Number of Corresponding phenotype values:\t%d' %df_pheno.shape[0])

    # select the samples id that we have in y_matrix.csv
    unique_ids_ymatrix = df_pheno['sample_ids'].unique()

    # filter the sample ids in x_matrix.csv that fits with the ids in y_matrix.csv
    df_genotypes = df_genotypes[df_genotypes['sample_ids'].isin(unique_ids_ymatrix)]

    # get the list of common ids between two datasets
    common_sample_ids = df_genotypes['sample_ids'].unique()

    # fileter again the sample ids in y_matrix.csv
    df_pheno = df_pheno[df_pheno['sample_ids'].isin(common_sample_ids)]

    # then map the continuous_values of y to x
    phenotype_dict_arr = df_pheno.set_index('sample_ids').to_dict()['pheno1']
    trans_phenotype_dict = {key: float(value) for key, value in phenotype_dict_arr.items()}
    df_genotypes['pheno1'] = df_genotypes['sample_ids'].map(trans_phenotype_dict) # add label column to genotypes data

    # create new X1, y1
    X1 = df_genotypes.iloc[:,1:df_genotypes.shape[1]-1]
    y1 = df_genotypes[['sample_ids', 'pheno1']]

    # print('------------------------------------------------------------------')
    # print(X1.iloc[:,1:].to_numpy())
    # print(y1['pheno1'].to_numpy())
    # print(y1.shape) #(500,2) for pheno1s
    # print('------------------------------------------------------------------\n')

    # convert new dataset to csv
    # print('------------------------------------------------------------------')
    # print('Convert original X y to new X y to csv files: ')
    X1.to_csv(datapath + '/data/x1_matrix.csv')
    y1.to_csv(datapath + '/data/y1_matrix.csv')
    # print('------------------------------------------------------------------\n')


def read_split_train_test_data(datapath):
    """
    Read the preprocessed data after matching input features and labels,
        + in: path to X and y
        + out: the X and y as type numpy array
    """
    X = pd.read_csv(datapath + '/data/x1_matrix.csv')
    y = pd.read_csv(datapath + '/data/y1_matrix.csv')

    X_nparray = X.iloc[:,2:]
    y_nparray = y.iloc[:,2]
    # print(X_nparray. shape, y_nparray.shape) #(500, 10000) (500,)

    X_train, X_test, y_train, y_test = train_test_split(X_nparray, y_nparray, train_size=0.9, shuffle=True)
    X_train.to_csv(datapath + '/data/x1_train_matrix.csv')
    y_train.to_csv(datapath + '/data/y1_train_matrix.csv')
    X_test.to_csv(datapath + '/data/x1_test_matrix.csv')
    y_test.to_csv(datapath + '/data/y1_test_matrix.csv')

# -------------------------------------------------------------
#  Prepare dataset
# -------------------------------------------------------------

# Standardize data
def standardize_data(X):
    standard_scaler = StandardScaler()
    standard_scaler.fit(X)
    X_scaled = standard_scaler.transform(X)
    return X_scaled

# Min Max Scaler
def minmax_scaler(y):
    minmax_scaler = MinMaxScaler()
    y = np.expand_dims(y, axis=1)
    y_scaled = minmax_scaler.fit_transform(y)
    return  y_scaled

# transform dataset to Tensor
def to_tensor(X, y):
    tensor_X = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    return tensor_X, tensor_y

# PCA
def decompose_PCA(X):
    pca = PCA(0.90)
    pca.fit(X)
    # print(pca.explained_variance_ratio_)
    # print(pca.components_)
    X = pca.transform(X)

    return X


def get_split_train_test(datapath):

    X_train = pd.read_csv(datapath + '/data/x1_train_matrix.csv')
    y_train = pd.read_csv(datapath + '/data/y1_train_matrix.csv')
    X_test = pd.read_csv(datapath + '/data/x1_test_matrix.csv')
    y_test = pd.read_csv(datapath + '/data/y1_test_matrix.csv')

    X_train_nparray, y_train_nparray = X_train.iloc[:,1:].to_numpy(), y_train.iloc[:,1].to_numpy()
    X_test_nparray, y_test_nparray = X_test.iloc[:,1:].to_numpy(), y_test.iloc[:,1].to_numpy()

    # Preprocessing dataset
    X_train_nparray = standardize_data(X_train_nparray)
    X_test_nparray = standardize_data(X_test_nparray)

    # X_scaled = X_nparray
    # y_scaled = minmax_scaler(y_nparray)
    # X_scaled = decompose_PCA(X_scaled)
    # print('X_scaled PCA:', X_scaled.shape)


    return X_train_nparray, y_train_nparray, X_test_nparray, y_test_nparray

