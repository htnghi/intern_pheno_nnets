import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import one_hot

# --------------------------------------------------------
# Read the raw data of genotype in .h5 format
# --------------------------------------------------------
with h5py.File("/Users/nghihuynh/Documents/MscTUM_BioTech/4th_semester/Internship/intern_pheno_nnets/src/data/genotype_data.h5", "r") as f:
    print('List of keys in the .h5 data file: ', list(f.keys()))
    # get data from the above keys
    snp_ids = f['snp_ids'][:].astype(str)
    sample_ids = f['sample_ids'][:].astype(int)
    X_012 = f['X_012'][:].astype(int)
    X_raw = f['X_raw'][:].astype(str)

print('Shape of X_raw: {}, type - {}'.format(X_raw.shape, type(X_raw)))
print('X_raw: ')
print(X_raw)
print('--------------------------------------------------------')
print('Shape of X_012: {}, type - {}'.format(X_012.shape, type(X_012)))
print('X_012 encoding:')
print(X_012)
print('--------------------------------------------------------')
print('Shape of sample_ids: ', sample_ids.shape)  #(2029,)
print('Sample IDs: ')
print(sample_ids)
print('--------------------------------------------------------')
print('SNP IDs or sample IDs:')
print( snp_ids)
print('--------------------------------------------------------\n')

# --------------------------------------------------------
# Write to csv file as a pandas dataframe for genotype data
# --------------------------------------------------------
arr_headers = np.insert(snp_ids, 0, 'sample_ids')
arr_data_X_raw = [list(arr_headers)]
arr_data_X_012 = [list(arr_headers)]
num_samples = len(X_raw)
for i in range(num_samples):
    row_Xraw = list(X_raw[i])
    sid = sample_ids[i]
    row_Xraw.insert(0, sid)
    arr_data_X_raw.append(row_Xraw)

    row_X012 = list(X_012[i])
    row_X012.insert(0, sid)
    arr_data_X_012.append(row_X012)

# convert to pandas dataframe
df_Xraw = pd.DataFrame(arr_data_X_raw[1:], columns=arr_data_X_raw[0])
df_X012 = pd.DataFrame(arr_data_X_012[1:], columns=arr_data_X_012[0])
# print(df_Xraw)
# print(df_X012)

# write to csv file
print('--------------------------------------------------------')
print('Convert dataset under .h5 file to .csv file:')
df_Xraw.to_csv('/Users/nghihuynh/Documents/MscTUM_BioTech/4th_semester/Internship/intern_pheno_nnets/src/data/raw_geneotype_dataset.csv')
df_X012.to_csv('/Users/nghihuynh/Documents/MscTUM_BioTech/4th_semester/Internship/intern_pheno_nnets/src/data/add012encoded_geneotype_dataset.csv')
print('--------------------------------------------------------\n')

# --------------------------------------------------------
# Read the phenotype data as the labels for training in .csv format
# --------------------------------------------------------
# df_phenotypes = pd.read_csv('/Users/nghihuynh/Documents/MscTUM_BioTech/4th_semester/Internship/intern_pheno_nnets/src/data/phenotype_data.csv', index_col=0)
# print('Phenotype data:')
# print(df_phenotypes)
# print('--------------------------------------------------------\n')
