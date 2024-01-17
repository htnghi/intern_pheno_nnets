import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# data1 = pd.read_csv('/Users/nghihuynh/Documents/MscTUM_BioTech/4th_semester/Internship/intern_pheno_nnets/src/data/pheno1/y_matrix_additive.csv')
# pheno1_label = data1['pheno1']
# sns.histplot(data=pheno1_label, stat='count',kde=True, bins=40, color='skyblue')
# plt.title('Histogram Distribution - Pheno1')
# plt.xlabel('Values')
# plt.ylabel('Count')
# plt.show()
# plt.savefig('histogram_pheno1.svg', bbox_inches='tight', dpi=600)
# plt.savefig('histogram_pheno1.pdf', bbox_inches='tight', dpi=600)

# Pheno2
# data2 = pd.read_csv('/Users/nghihuynh/Documents/MscTUM_BioTech/4th_semester/Internship/intern_pheno_nnets/src/data/pheno2/y_matrix_additive.csv')
# pheno2_label = data2['pheno2']
# sns.histplot(data=pheno2_label, stat='count',kde=True, bins=40, color='skyblue')
# plt.title('Histogram Distribution - Pheno2')
# plt.xlabel('Values')
# plt.ylabel('Count')
# plt.show()
# plt.savefig('histogram_pheno2.svg', bbox_inches='tight', dpi=600)
# plt.savefig('histogram_pheno2.pdf', bbox_inches='tight', dpi=600)

# # Pheno3
data3 = pd.read_csv('/Users/nghihuynh/Documents/MscTUM_BioTech/4th_semester/Internship/intern_pheno_nnets/src/data/pheno3/y_matrix_additive.csv')
pheno3_label = data3['pheno3']
sns.histplot(data=pheno3_label, stat='count',kde=True, bins=40, color='skyblue')
plt.title('Histogram Distribution - Pheno3')
plt.xlabel('Values')
plt.ylabel('Count')
# plt.show()
plt.savefig('histogram_pheno3.svg', bbox_inches='tight', dpi=600)
plt.savefig('histogram_pheno3.pdf', bbox_inches='tight', dpi=600)