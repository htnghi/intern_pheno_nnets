import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data by pandas
df_genotypes = pd.read_csv('x_matrix.csv')
df_phenotypes = pd.read_csv('y_matrix.csv')

# Some basis summary about the dataset
print('Number of Samples:\t%d' %df_genotypes.shape[0]) #2029
print('Number of Features:\t%d' %(df_genotypes.shape[1]-1)) #2000
print('Number of Corresponding phenotype values:\t%d' %df_phenotypes.shape[0]) #1163
print('---------------------------------------------')

# Mapping orders samples of genotypes to phenotypes
phenotype_dict_arr = df_phenotypes.set_index('sample_id').to_dict()['continuous_values']
df_genotypes['biomass'] = df_genotypes['sample_id'].map(phenotype_dict_arr) # add label column to genotypes data
X = df_genotypes.iloc[:,1:df_genotypes.shape[1]-1].to_numpy()
y = df_genotypes.iloc[:,df_genotypes.shape[1]-1].to_numpy()

print('Shape of X: \t', X.shape)
print(X)
print('Shape of y: \t', y.shape)
print(y)

#df_phenotypes.dropna()
common_samples = df_phenotypes['sample_id'].unique()

# Filter both DataFrames to retain only common samples
df_genotypes = df_genotypes[df_genotypes['sample_id'].isin(common_samples)]
df_phenotypes = df_phenotypes[df_phenotypes['sample_id'].isin(common_samples)]
X1 = df_genotypes.iloc[:,1:df_genotypes.shape[1]-1].to_numpy()
y1 = df_genotypes.iloc[:,df_genotypes.shape[1]-1].to_numpy()
print('Shape of X1: \t', X1.shape)
print(X1)
print('Shape of y1: \t', y1.shape)
print(y1)

print('count nan value of genotype: \t',df_genotypes.isnull().sum().sum())
print('count nan value of phenotype: \t',df_phenotypes.isnull().sum().sum())
print(df_phenotypes.isnull().any(axis=1))