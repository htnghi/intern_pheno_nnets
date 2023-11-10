import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import one_hot

# Read the prepared (raw) data by pandas
def read_data(datapath):
    """
    Read the raw data or impured data by pandas,
        + in: temporarily the data path is fixed
        + out: the returned format is pandas dataframe and write to .csv files
    """

    df_genotypes  = pd.read_csv(datapath + '/add012encoded_geneotype_dataset.csv')
    df_phenotypes = pd.read_csv(datapath + '/phenotype_data.csv')

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
    X1.to_csv(datapath + '/x1_matrix.csv')
    y1.to_csv(datapath + '/y1_matrix.csv')
    # print('------------------------------------------------------------------\n')


def read_prerocessed_data(datapath):
    """
    Read the preprocessed data after matching input features and labels,
        + in: path to X and y
        + out: the X and y as type numpy array
    """
    X = pd.read_csv(datapath + '/x1_matrix.csv')
    y = pd.read_csv(datapath + '/y1_matrix.csv')

    X_nparray = X.iloc[:,2:].to_numpy()
    y_nparray = y.iloc[:,2].to_numpy()
    # print(X_nparray. shape, y_nparray.shape) #(500, 10000) (500,)

    return X_nparray, y_nparray

