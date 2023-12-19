import pandas as pd
import numpy as np

import torch
from torch.nn.functional import one_hot

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------
#  One_hot encoding
# -------------------------------------------------------------
def get_onehot_encoding(X: np.array):
    """
    Genotype matrix is homozygous => create 3d torch tensor with (samples, SNPs, 4), with 4 as the onehot encoding
    - A : [1,0,0,0]
    - C : [0,1,0,0]
    - G : [0,0,1,0]
    - T : [0,0,0,1]
    """

    unique, inverse = np.unique(X, return_inverse=True)
    inverse = inverse.reshape(X.shape)
    X_onehot = one_hot(torch.from_numpy(inverse)).numpy()
    return X_onehot


def read_data_pheno_onehot(datapath, type):
    """
    Read the raw data or impured data by pandas,
        + in: temporarily the data path is fixed
        + out: the returned format is pandas dataframe and write to .csv files
    """

    df_genotypes  = pd.read_csv(datapath + '/data/raw_geneotype_dataset.csv')
    df_phenotypes = pd.read_csv(datapath + '/data/phenotype_data.csv')

    # delete nan values from phenotye data
    df_pheno = df_phenotypes.dropna(subset=['sample_ids', 'pheno'+str(type)]) #only delte missing value in pheno1 column
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
    phenotype_dict_arr = df_pheno.set_index('sample_ids').to_dict()['pheno'+str(type)]
    trans_phenotype_dict = {key: float(value) for key, value in phenotype_dict_arr.items()}
    df_genotypes['pheno'+str(type)] = df_genotypes['sample_ids'].map(trans_phenotype_dict) # add label column to genotypes data

    # create new X1, y1
    X = df_genotypes.iloc[:,1:df_genotypes.shape[1]-1]
    y = df_genotypes[['sample_ids', 'pheno'+str(type)]]

    # convert new dataset to csv
    X.to_csv(datapath + '/data/pheno' + str(type) + '/x_matrix_onehot.csv')
    y.to_csv(datapath + '/data/pheno' + str(type) + '/y_matrix_onehot.csv')
    # print('------------------------------------------------------------------\n')


def split_train_test_data_onehot(datapath, type):
    """
    Read the preprocessed data after matching input features and labels,
        + in: path to X and y
        + out: the X and y as type numpy array
    """
    
    X = pd.read_csv(datapath + '/data/pheno' + str(type) + '/x_matrix_onehot.csv')
    y = pd.read_csv(datapath + '/data/pheno' + str(type) + '/y_matrix_onehot.csv')

    X_nparray = X.iloc[:,2:]
    y_nparray = y.iloc[:,2]

    X_train, X_test, y_train, y_test = train_test_split(X_nparray, y_nparray, train_size=0.9, shuffle=True)
    
    X_train.to_csv(datapath + '/data/pheno' + str(type) + '/x_train_onehot.csv')
    y_train.to_csv(datapath + '/data/pheno' + str(type) + '/y_train_onehot.csv')
    X_test.to_csv(datapath + '/data/pheno' + str(type) + '/x_test_onehot.csv')
    y_test.to_csv(datapath + '/data/pheno' + str(type) + '/y_test_onehot.csv')


def load_split_train_test_onehot(datapath, type):

    X_train = pd.read_csv(datapath + '/data/pheno' + str(type) + '/x_train_onehot.csv')
    y_train = pd.read_csv(datapath + '/data/pheno' + str(type) + '/y_train_onehot.csv')
    X_test = pd.read_csv(datapath + '/data/pheno' + str(type) + '/x_test_onehot.csv')
    y_test = pd.read_csv(datapath + '/data/pheno' + str(type) + '/y_test_onehot.csv')

    X_train_nparray, y_train_nparray = X_train.iloc[:,1:].to_numpy(), y_train.iloc[:,1].to_numpy()
    X_test_nparray, y_test_nparray = X_test.iloc[:,1:].to_numpy(), y_test.iloc[:,1].to_numpy()

    X_train_nparray = get_onehot_encoding(X_train_nparray)
    X_test_nparray = get_onehot_encoding(X_test_nparray)

    return X_train_nparray, y_train_nparray, X_test_nparray, y_test_nparray


# -------------------------------------------------------------
#  Additive Encoding
# -------------------------------------------------------------
# Read the prepared (raw) data by pandas
def read_data_pheno_additive(datapath, type):
    """
    Read the raw data or impured data by pandas,
        + in: temporarily the data path is fixed
        + out: the returned format is pandas dataframe and write to .csv files
    """

    df_genotypes  = pd.read_csv(datapath + '/data/add012encoded_geneotype_dataset.csv')
    df_phenotypes = pd.read_csv(datapath + '/data/phenotype_data.csv')

    # delete nan values from phenotye data
    df_pheno = df_phenotypes.dropna(subset=['sample_ids', 'pheno'+str(type)]) #only delte missing value in pheno1 column
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
    phenotype_dict_arr = df_pheno.set_index('sample_ids').to_dict()['pheno'+str(type)]
    trans_phenotype_dict = {key: float(value) for key, value in phenotype_dict_arr.items()}
    df_genotypes['pheno'+str(type)] = df_genotypes['sample_ids'].map(trans_phenotype_dict) # add label column to genotypes data

    # create new X1, y1
    X = df_genotypes.iloc[:,1:df_genotypes.shape[1]-1]
    y = df_genotypes[['sample_ids', 'pheno'+str(type)]]

    # convert new dataset to csv
    X.to_csv(datapath + '/data/pheno' + str(type) + '/x_matrix_additive.csv')
    y.to_csv(datapath + '/data/pheno' + str(type) + '/y_matrix_additive.csv')
    # print('------------------------------------------------------------------\n')

def split_train_test_data_additive(datapath, type):
    """
    Read the preprocessed data after matching input features and labels,
        + in: path to X and y
        + out: the X and y as type numpy array
    """
    X = pd.read_csv(datapath + '/data/pheno' + str(type) + '/x_matrix_additive.csv')
    y = pd.read_csv(datapath + '/data/pheno' + str(type) + '/y_matrix_additive.csv')

    X_nparray = X.iloc[:,2:]
    y_nparray = y.iloc[:,2]

    X_train, X_test, y_train, y_test = train_test_split(X_nparray, y_nparray, train_size=0.9, shuffle=True)
    X_train.to_csv(datapath + '/data/pheno' + str(type) + '/x_train_additive.csv')
    y_train.to_csv(datapath + '/data/pheno' + str(type) + '/y_train_additive.csv')
    X_test.to_csv(datapath + '/data/pheno' + str(type) + '/x_test_additive.csv')
    y_test.to_csv(datapath + '/data/pheno' + str(type) + '/y_test_additive.csv')

def load_split_train_test_additive(datapath, type):

    X_train = pd.read_csv(datapath + '/data/pheno' + str(type) + '/x_train_additive.csv')
    y_train = pd.read_csv(datapath + '/data/pheno' + str(type) + '/y_train_additive.csv')
    X_test = pd.read_csv(datapath + '/data/pheno' + str(type) + '/x_test_additive.csv')
    y_test = pd.read_csv(datapath + '/data/pheno' + str(type) + '/y_test_additive.csv')

    X_train_nparray, y_train_nparray = X_train.iloc[:,1:].to_numpy(), y_train.iloc[:,1].to_numpy()
    X_test_nparray, y_test_nparray = X_test.iloc[:,1:].to_numpy(), y_test.iloc[:,1].to_numpy()

    return X_train_nparray, y_train_nparray, X_test_nparray, y_test_nparray