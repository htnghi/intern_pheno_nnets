import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


data1 = pd.read_csv('/Users/nghihuynh/Documents/MscTUM_BioTech/4th_semester/Internship/intern_pheno_nnets/src/utils/plot_preprocessing/result_data_preprocessing.csv')


fig, ax = plt.subplots(1, 1, figsize=(12, 7))
sns.barplot(x = 'Technique', y = 'Explain_Variance', hue = 'phenotype', data = data1, palette = 'Spectral', ax=ax)
plt.xlabel('Data Preprocessing Technique')
plt.ylabel('Explain Variance')
plt.show()



