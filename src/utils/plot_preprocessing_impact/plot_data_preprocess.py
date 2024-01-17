import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# set theme for figure
sns.set_theme(style="whitegrid")

# read the data
dataframe_mlp = pd.read_csv('./mlp_result_data_preprocessing_impacts.csv')
dataframe_cnn = pd.read_csv('./cnn_result_data_preprocessing_impacts.csv')
dataframe_rnn = pd.read_csv('./rnn_result_data_preprocessing_impacts.csv')

# get mlp dataframe for each type of preprocessing
df_mlp_nonprocess = dataframe_mlp[(dataframe_mlp.Options == 'non_preproc')]
df_mlp_yscaled = dataframe_mlp[(dataframe_mlp.Options == 'y_scaled')]
df_mlp_xnorm_yscaled = dataframe_mlp[(dataframe_mlp.Options == 'X_norm+y_scaled')]
df_mlp_xnorm_yscaled_pca = dataframe_mlp[(dataframe_mlp.Options == 'X_norm+PCA+y_scaled')]

# get cnn dataframe for each type of preprocessing
df_cnn_nonprocess = dataframe_cnn[(dataframe_cnn.Options == 'non_preproc')]
df_cnn_yscaled = dataframe_cnn[(dataframe_cnn.Options == 'y_scaled')]

# get rnn dataframe for each type of preprocessing
df_rnn_nonprocess = dataframe_rnn[(dataframe_rnn.Options == 'non_preproc')]
df_rnn_yscaled = dataframe_rnn[(dataframe_rnn.Options == 'y_scaled')]

# plot the figures
gs = gridspec.GridSpec(2,4)
fig = plt.figure(figsize=(18,8))
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])
ax4 = plt.subplot(gs[0,3])
ax5 = plt.subplot(gs[1,0])
ax6 = plt.subplot(gs[1,1])
ax7 = plt.subplot(gs[1,2])
ax8 = plt.subplot(gs[1,3])


# plot the mlp figure by seaborn
barchart1 = sns.barplot(x='Dataset', y='Expvar', hue='Experiment', data=df_mlp_nonprocess, palette='Spectral', ax=ax1)
barchart1.set(xlabel=None, ylabel="ExpVar", title='MLP: non_preprocessing')
ax1.set_ylim([0.0, 1.0])
ax1.legend(loc='upper left')

barchart2 = sns.barplot(x='Dataset', y='Expvar', hue='Experiment', data=df_mlp_yscaled, palette='Spectral', ax=ax2)
barchart2.set(xlabel=None, ylabel=None, title='MLP: y_scaled')
ax2.set_ylim([0.0, 1.0])
ax2.legend(loc='upper left')

barchart3 = sns.barplot(x='Dataset', y='Expvar', hue='Experiment', data=df_mlp_xnorm_yscaled, palette='Spectral', ax=ax3)
barchart3.set(xlabel=None, ylabel=None, title='MLP: X_norm+y_scaled')
ax3.set_ylim([0.0, 1.0])
ax3.legend(loc='upper left')

barchart4 = sns.barplot(x='Dataset', y='Expvar', hue='Experiment', data=df_mlp_xnorm_yscaled_pca, palette='Spectral', ax=ax4)
barchart4.set(xlabel=None, ylabel=None, title='MLP: X_norm+PCA+y_scaled')
ax4.set_ylim([0.0, 1.0])
ax4.legend(loc='upper left')

# plot the cnn figure by seaborn
barchart5 = sns.barplot(x='Dataset', y='Expvar', hue='Experiment', data=df_cnn_nonprocess, palette='Spectral', ax=ax5)
barchart5.set(xlabel=None, ylabel="ExpVar", title='CNN: non_preprocessing')
ax5.set_ylim([-0.4, 1.0])
ax5.legend(loc='lower right')

barchart6 = sns.barplot(x='Dataset', y='Expvar', hue='Experiment', data=df_cnn_yscaled, palette='Spectral', ax=ax6)
barchart6.set(xlabel=None, ylabel=None, title='CNN: y_scaled')
ax6.set_ylim([-0.4, 1.0])
ax6.legend(loc='lower right')

# plot the rnn figure by seaborn
barchart7 = sns.barplot(x='Dataset', y='Expvar', hue='Experiment', data=df_rnn_nonprocess, palette='Spectral', ax=ax7)
barchart7.set(xlabel=None, ylabel=None, title='RNN: non_preprocessing')
ax7.set_ylim([-1.2, 0.2])
ax7.legend(loc='lower right')

barchart8 = sns.barplot(x='Dataset', y='Expvar', hue='Experiment', data=df_rnn_yscaled, palette='Spectral', ax=ax8)
barchart8.set(xlabel=None, ylabel=None, title='RNN: y_scaled')
ax8.set_ylim([-1.2, 0.2])
ax8.legend(loc='lower right')

plt.savefig('barchart_' + 'preprocessing' + '.pdf', bbox_inches='tight')
