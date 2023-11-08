import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import one_hot

# Read the raw data of genotype (h5py)
with h5py.File("/Users/nghihuynh/Documents/MscTUM_BioTech/4th_semester/Internship/intern_pheno_nnets/src/data/genotype_data.h5", "r") as f:
    print(list(f.keys()))
    snp_ids = f['snp_ids'][:].astype(str)
    sample_ids = f['sample_ids'][:].astype(int)
    X = f['X_012'][:].astype(int)
    #X_012 = np.array(f['X_012'])
    X_raw = f['X_raw'][:].astype(str)


print('X_raw:\t\t', X_raw.shape)
print('Genotypes data (raw): \n', X_raw)
print('Shape of X (after 012 encoding):\t\t', X.shape)
print('Genotypes data after 012 encoding: \n', X)
print('sample_ids:\t', sample_ids.shape)  #(2029,)
print('snp_ids:\t', snp_ids.shape)


# Read the raw data of phenotype (csv) by pandas
df_phenotypes = pd.read_csv('/Users/nghihuynh/Documents/MscTUM_BioTech/4th_semester/Internship/intern_pheno_nnets/src/data/phenotype_data.csv', index_col=0)
#df_phenotypes.head() #show the first 5 rows
# print('Number of raw phenotype values:\t%d' %df_phenotypes.shape[0])
# print('Phenotypes data: \n', df_phenotypes)
