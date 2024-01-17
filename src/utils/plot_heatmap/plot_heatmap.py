import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import sys
import glob
import pprint
import warnings

from matplotlib.patches import Rectangle
from matplotlib_venn import venn3, venn3_circles, venn3_unweighted

# configure matplotlib
matplotlib.style.use('ggplot')
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 12
# plt.rc('text.latex', preamble=r'\usepackage{lmodern}')

# read data
if len(sys.argv) < 2:
    print("Error: missing arguments!")
    print("Usage: python plot_heatmap.py <file_csv>")
    exit(1)

filename = sys.argv[1]
df_results = pd.read_csv(filename)
print('Dataframe read from the input file:')
print(df_results)
print('------------------------------------------------------')

# ------------------------------------------------------
# plot the figure
# ------------------------------------------------------
fig, ax = plt.subplots(figsize=(9,3))

# labels and create a new dataframe for plotting


# extract the values of columns that read from df_results_overview with the regex='mean'
plot_data = df_results.iloc[:,1:]
label_df = plot_data

print(plot_data)

# plot the heatmap
sns.heatmap(data=plot_data, cmap="Spectral", cbar_kws={"shrink": .75}, vmin=-0.25, vmax=0.65,
            annot=label_df, fmt='', linewidths=1.5, linecolor='white', cbar=True, annot_kws={"size":12})

# adjust the axes
ax.set_xticklabels(label_df.columns, rotation=0)
ax.set_yticklabels(df_results['phenotype'], rotation=0)
ax.tick_params(top=False,
               bottom=False,
               left=False,
               right=False,
               labelleft=True,
               labelbottom=True)

# optional: for adding the pads
# for row, index in enumerate(plot_data.index):
#     position = label_df.columns.get_loc(row_max[index].split('_')[0])
#     # print('row[{}], index[{}]: position={}'.format(row, index, position))
#     ax.add_patch(Rectangle((position, row), 1, 1, fill=False, edgecolor='0', lw=1.5))

# save to file or display the figure
fig.tight_layout()
# plt.show()
plt.savefig('heatmap_' + filename.split('.')[0] + '.svg', bbox_inches='tight', dpi=600)
