import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import glob
import pprint
import warnings
import sys
import re

from matplotlib.patches import Rectangle
from matplotlib_venn import venn3, venn3_circles, venn3_unweighted

import matplotlib.gridspec as gridspec

# read csv data with pandas
data = pd.read_csv('./data_optuna_tuning_mlp_history.csv') 

# extract and mask the data
nparr_mlp_expvar_data1_expvals = data['mlp_expvarobj_data1_expvals'].to_numpy().astype(np.double)
masks_mlp_expvar_data1_expvals = np.isfinite(nparr_mlp_expvar_data1_expvals)
nparr_mlp_mse_data1_lossvals = data['mlp_mselossobj_data1_msevals'].to_numpy().astype(np.double)
masks_mlp_mse_data1_lossvals = np.isfinite(nparr_mlp_mse_data1_lossvals)
nparr_mlp_mse_data1_expvals = data['mlp_mselossobj_data1_expvals'].to_numpy().astype(np.double)


nparr_mlp_expvar_data2_expvals = data['mlp_expvarobj_data2_expvals'].to_numpy().astype(np.double)
masks_mlp_expvar_data2_expvals = np.isfinite(nparr_mlp_expvar_data2_expvals)
nparr_mlp_mse_data2_lossvals = data['mlp_mselossobj_data2_msevals'].to_numpy().astype(np.double)
masks_mlp_mse_data2_lossvals = np.isfinite(nparr_mlp_mse_data2_lossvals)
nparr_mlp_mse_data2_expvals = data['mlp_mselossobj_data2_expvals'].to_numpy().astype(np.double)


nparr_mlp_expvar_data3_expvals = data['mlp_expvarobj_data3_expvals'].to_numpy().astype(np.double)
masks_mlp_expvar_data3_expvals = np.isfinite(nparr_mlp_expvar_data3_expvals)
nparr_mlp_mse_data3_lossvals = data['mlp_mselossobj_data3_msevals'].to_numpy().astype(np.double)
masks_mlp_mse_data3_lossvals = np.isfinite(nparr_mlp_mse_data3_lossvals)
nparr_mlp_mse_data3_expvals = data['mlp_mselossobj_data3_expvals'].to_numpy().astype(np.double)

# plot the figures
gs = gridspec.GridSpec(1,3)
fig = plt.figure(figsize=(15,5))
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])

# for x-axis data
x = np.arange(0, 100, 1)

# ----------------------------------------------
# 1st figure
# ----------------------------------------------

# annotate the text for best trial in expvar obj direction
best_data1_expvar_val = max(nparr_mlp_expvar_data1_expvals)
best_data1_expvar_idx = np.where(nparr_mlp_expvar_data1_expvals==best_data1_expvar_val)[0][0]
xloc_best_data1_expvar = x[best_data1_expvar_idx]
annot_best_expvar_data1_text = "Best Trial: {:d}\n".format(xloc_best_data1_expvar)
annot_best_expvar_data1_text += "Expvar (Validation): {:.3f}\n".format(best_data1_expvar_val)
annot_best_expvar_data1_text += "Expvar (Independent Test): {:.3f}\n".format(0.4435)

# plot the 1st figure in direction of expvar obj
ax1.set_xlabel('Trials (Dataset: Pheno1)')
ax1.set_ylabel('expvar', color='tab:red')
ax1.plot(x[masks_mlp_expvar_data1_expvals], nparr_mlp_expvar_data1_expvals[masks_mlp_expvar_data1_expvals], color='tab:red', marker='o')
ax1.annotate(annot_best_expvar_data1_text, xy=(xloc_best_data1_expvar, best_data1_expvar_val), 
            xytext=(xloc_best_data1_expvar-45, best_data1_expvar_val-0.95),
            arrowprops=dict(arrowstyle="fancy"))
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.set_ylim([-3.0, 1.0])
ax1.grid()

# annotate the text for best trial in mseloss obj direction
best_data1_mseloss_val = min(nparr_mlp_mse_data1_lossvals)
best_data1_mseloss_idx = np.where(nparr_mlp_mse_data1_lossvals==best_data1_mseloss_val)[0][0]
best_data1_mseloss_expvar = nparr_mlp_mse_data1_expvals[best_data1_mseloss_idx]
xloc_best_data1_mseloss = x[best_data1_mseloss_idx]
annot_best_mse_data1_text = "Best Trial: {:d}\n".format(best_data1_mseloss_idx)
annot_best_mse_data1_text += "Expvar (Validation): {:.3f}\n".format(best_data1_mseloss_expvar)
annot_best_mse_data1_text += "Expvar (Independent Test): {:.3f}\n".format(0.5664)

# plot the 1st figure in direction of mseloss obj
ax1_2nd_yaxis = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax1_2nd_yaxis.set_ylabel('mseloss', color='tab:blue')
ax1_2nd_yaxis.plot(x[masks_mlp_mse_data1_lossvals], nparr_mlp_mse_data1_lossvals[masks_mlp_mse_data1_lossvals], color='tab:blue', marker='o')
ax1_2nd_yaxis.annotate(annot_best_mse_data1_text, xy=(xloc_best_data1_mseloss, best_data1_mseloss_val), 
            xytext=(xloc_best_data1_mseloss-65, best_data1_mseloss_val+0.05),
            arrowprops=dict(arrowstyle="fancy"))
ax1_2nd_yaxis.tick_params(axis='y', labelcolor='tab:blue')
ax1_2nd_yaxis.set_ylim([0.0, 0.25])

# ----------------------------------------------
# 2nd figure
# ----------------------------------------------

# annotate the text for best trial in expvar obj direction
best_data2_expvar_val = max(nparr_mlp_expvar_data2_expvals)
best_data2_expvar_idx = np.where(nparr_mlp_expvar_data2_expvals==best_data2_expvar_val)[0][0]
xloc_best_data2_expvar = x[best_data2_expvar_idx]
annot_best_expvar_data2_text = "Best Trial: {:d}\n".format(xloc_best_data2_expvar)
annot_best_expvar_data2_text += "Expvar (Validation): {:.3f}\n".format(best_data2_expvar_val)
annot_best_expvar_data2_text += "Expvar (Independent Test): {:.3f}\n".format(0.4851)

ax2.set_xlabel('Trials (Dataset: Pheno2)')
# ax2.set_ylabel('expvar', color='tab:red')
ax2.plot(x[masks_mlp_expvar_data2_expvals], nparr_mlp_expvar_data2_expvals[masks_mlp_expvar_data2_expvals], color='tab:red', marker='o')
ax2.annotate(annot_best_expvar_data2_text, xy=(xloc_best_data2_expvar, best_data2_expvar_val), 
            xytext=(xloc_best_data2_expvar-55, best_data2_expvar_val-0.85),
            arrowprops=dict(arrowstyle="fancy"))
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim([-3.0, 1.0])
ax2.grid()

# annotate the text for best trial in mseloss obj direction
best_data2_mseloss_val = min(nparr_mlp_mse_data2_lossvals)
best_data2_mseloss_idx = np.where(nparr_mlp_mse_data2_lossvals==best_data2_mseloss_val)[0][0]
best_data2_mseloss_expvar = nparr_mlp_mse_data2_expvals[best_data2_mseloss_idx]
xloc_best_data2_mseloss = x[best_data2_mseloss_idx]
annot_best_mse_data2_text = "Best Trial: {:d}\n".format(best_data2_mseloss_idx)
annot_best_mse_data2_text += "Expvar (Validation): {:.3f}\n".format(best_data2_mseloss_expvar)
annot_best_mse_data2_text += "Expvar (Independent Test): {:.3f}\n".format(0.4682)

ax2_2nd_yaxis = ax2.twinx()  # instantiate a second axes that shares the same x-axis
# ax2_2nd_yaxis.set_ylabel('mseloss', color='tab:blue') 
ax2_2nd_yaxis.plot(x[masks_mlp_mse_data2_lossvals], nparr_mlp_mse_data2_lossvals[masks_mlp_mse_data2_lossvals], color='tab:blue', marker='o')
ax2_2nd_yaxis.annotate(annot_best_mse_data2_text, xy=(xloc_best_data2_mseloss, best_data2_mseloss_val), 
            xytext=(xloc_best_data2_mseloss-40, best_data2_mseloss_val+0.05),
            arrowprops=dict(arrowstyle="fancy"))
ax2_2nd_yaxis.tick_params(axis='y', labelcolor='tab:blue')
ax2_2nd_yaxis.set_ylim([0.0, 0.25])

# ----------------------------------------------
# 3rd figure
# ----------------------------------------------

# annotate the text for best trial in expvar obj direction
best_data3_expvar_val = max(nparr_mlp_expvar_data3_expvals)
best_data3_expvar_idx = np.where(nparr_mlp_expvar_data3_expvals==best_data3_expvar_val)[0][0]
xloc_best_data3_expvar = x[best_data3_expvar_idx]
annot_best_expvar_data3_text = "Best Trial: {:d}\n".format(xloc_best_data3_expvar)
annot_best_expvar_data3_text += "Expvar (Validation): {:.3f}\n".format(best_data3_expvar_val)
annot_best_expvar_data3_text += "Expvar (Independent Test): {:.3f}\n".format(0.6974)

ax3.set_xlabel('Trials (Dataset: Pheno3)')
# ax3.set_ylabel('expvar', color='tab:red')
ax3.plot(x[masks_mlp_expvar_data3_expvals], nparr_mlp_expvar_data3_expvals[masks_mlp_expvar_data3_expvals], color='tab:red', marker='o')
ax3.annotate(annot_best_expvar_data3_text, xy=(xloc_best_data3_expvar, best_data3_expvar_val), 
            xytext=(xloc_best_data3_expvar+2, best_data3_expvar_val-0.85),
            arrowprops=dict(arrowstyle="fancy"))
ax3.tick_params(axis='y', labelcolor='tab:red')
ax3.set_ylim([-3.0, 1.0])
ax3.grid()

# annotate the text for best trial in mseloss obj direction
best_data3_mseloss_val = min(nparr_mlp_mse_data3_lossvals)
best_data3_mseloss_idx = np.where(nparr_mlp_mse_data3_lossvals==best_data3_mseloss_val)[0][0]
best_data3_mseloss_expvar = nparr_mlp_mse_data3_expvals[best_data3_mseloss_idx]
xloc_best_data3_mseloss = x[best_data3_mseloss_idx]
annot_best_mse_data3_text = "Best Trial: {:d}\n".format(best_data3_mseloss_idx)
annot_best_mse_data3_text += "Expvar (Validation): {:.3f}\n".format(best_data3_mseloss_expvar)
annot_best_mse_data3_text += "Expvar (Independent Test): {:.3f}\n".format(0.7075)

ax3_2nd_yaxis = ax3.twinx()  # instantiate a second axes that shares the same x-axis
ax3_2nd_yaxis.set_ylabel('mseloss', color='tab:blue') 
ax3_2nd_yaxis.plot(x[masks_mlp_mse_data3_lossvals], nparr_mlp_mse_data3_lossvals[masks_mlp_mse_data3_lossvals], color='tab:blue', marker='o')
ax3_2nd_yaxis.annotate(annot_best_mse_data3_text, xy=(xloc_best_data3_mseloss, best_data3_mseloss_val), 
            xytext=(xloc_best_data3_mseloss-2, best_data3_mseloss_val+0.05),
            arrowprops=dict(arrowstyle="fancy"))
ax3_2nd_yaxis.tick_params(axis='y', labelcolor='tab:blue')
ax3_2nd_yaxis.set_ylim([0.0, 0.25])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('./optuna_tuning_history_mlp.pdf', bbox_inches='tight')
# plt.show()