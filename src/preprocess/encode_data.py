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

    df_genotypes  = pd.read_csv(datapath + '/x_matrix.csv')
    df_phenotypes = pd.read_csv(datapath + '/y_matrix.csv')

    # Some basis summary about the dataset
    print('------------------------------------------------------------------')
    print('Number of Samples:\t%d' %df_genotypes.shape[0]) #2029
    print('Number of Features:\t%d' %(df_genotypes.shape[1]-1)) #2000
    print('Number of Corresponding phenotype values:\t%d' %df_phenotypes.shape[0]) #1163
    print('------------------------------------------------------------------\n')

    # select the samples id that we have in y_matrix.csv
    unique_ids_ymatrix = df_phenotypes['sample_id'].unique()

    # filter the sample ids in x_matrix.csv that fits with the ids in y_matrix.csv
    df_genotypes = df_genotypes[df_genotypes['sample_id'].isin(unique_ids_ymatrix)]

    # get the list of common ids between two datasets
    common_sample_ids = df_genotypes['sample_id'].unique()

    # fileter again the sample ids in y_matrix.csv
    df_phenotypes = df_phenotypes[df_phenotypes['sample_id'].isin(common_sample_ids)]

    # then map the continuous_values of y to x
    phenotype_dict_arr = df_phenotypes.set_index('sample_id').to_dict()['multiclass_values']
    trans_phenotype_dict = {key: int(value)-1 for key, value in phenotype_dict_arr.items()}
    df_genotypes['biomass'] = df_genotypes['sample_id'].map(trans_phenotype_dict) # add label column to genotypes data

    # create new X1, y1
    X1 = df_genotypes.iloc[:,0:df_genotypes.shape[1]-1]
    y1 = df_genotypes[['sample_id', 'biomass']]

    print('------------------------------------------------------------------')
    print(X1.iloc[:,1:].to_numpy())
    print(y1['biomass'].to_numpy())
    print('------------------------------------------------------------------\n')

    # convert new dataset to csv
    print('------------------------------------------------------------------')
    print('Convert original X y to new X y to csv files: ')
    X1.to_csv(datapath + '/x1_matrix.csv')
    y1.to_csv(datapath + '/y1_matrix.csv')
    print('------------------------------------------------------------------\n')


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

    return X_nparray, y_nparray

def get_onehot_encoding(X):
    """
    Genotype matrix is homozygous => Shape torch tensor (samples, SNPs, 4).
    """
    unique, inverse = np.unique(X, return_inverse=True)
    inverse = inverse.reshape(X.shape)
    X_onehot = one_hot(torch.from_numpy(inverse)).numpy()

    return X_onehot

def get_additive_encoding(X: np.array, style: str = '012') -> np.array:
    """
    'Style 012' 0: homo major allele, 1: hetero, 2: homo minor allele
    """
    alleles = []
    index_arr = []
    pairs = [['A', 'C'], ['A', 'G'], ['A', 'T'], ['C', 'G'], ['C', 'T'], ['G', 'T']]
    heterozygous_nuc = ['M', 'R', 'W', 'S', 'Y', 'K']
    for j, col in enumerate(np.transpose(X)):
        unique, inv, counts = np.unique(col, return_counts=True, return_inverse=True)
        unique = unique.astype(str)
        boolean = (unique == 'A') | (unique == 'T') | (unique == 'C') | (unique == 'G')
        tmp = np.zeros(3)
        if len(unique) > 3:
            raise Exception('More than two alleles encountered at snp ' + str(j))
        elif len(unique) == 3:
            hetero = unique[~boolean][0]
            homozygous = unique[boolean]
            for i, pair in enumerate(pairs):
                if all(h in pair for h in homozygous) and hetero != heterozygous_nuc[i]:
                    raise Exception('More than two alleles encountered at snp ' + str(i))
            tmp[~boolean] = 1.0 
            tmp[np.argmin(counts[boolean])] = 2.0 
        elif len(unique) == 2:
            if list(unique) in pairs:
                tmp[np.argmin(counts)] = 2.0 
            else:
                tmp[(~boolean).nonzero()] = 1.0 
        else:
            if unique[0] in heterozygous_nuc:
                tmp[0] = 1.0 
        alleles.append(tmp)
        index_arr.append(inv)
    alleles = np.transpose(np.array(alleles))
    ind_arr = np.transpose(np.array(index_arr))
    cols = np.arange(alleles.shape[1])
    return alleles[ind_arr, cols]



# load X, y preprocessed
# print('------------------------------------------------------------------')
# print('Read the preprocessed data: showing X')
# X = pd.read_csv('../data/x1_matrix.csv')
# y = pd.read_csv('../data/y1_matrix.csv')
# X_nparray = X.iloc[:,2:].to_numpy()
# print(X_nparray)

# print('------------------------------------------------------------------')
# print('Check one-hot encoding: ')
# X_onehot_encoded = get_onehot_encoding(X_nparray)
# print('Shape X_onehot_encoded: ', X_onehot_encoded.shape)
# print(X_onehot_encoded[0])

# print('Check additive encoding: ')
# X_additive_encoded = get_additive_encoding(X_nparray, '012')
# print('Shape X_additive_encoded: ', X_additive_encoded.shape)
# print(X_additive_encoded[0])
# print('------------------------------------------------------------------\n')

# df_atwll = pd.read_hdf('../data/atwell_ld_pruned.h5')
# print(df_atwll)