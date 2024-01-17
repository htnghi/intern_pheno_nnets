## Deep Nerual Networks for Phenotype Prediction

This project is about trying and testing deep neural networks for predicting phenotype from genotype datasets (SNPs dataset). The target is a comparison between deep neural network models in phenotype prediction.

### Overview - implemented models
So far, have done the implementation of MLP, CNN, RNN with Pytorch. All models are tuned using Optuna.

- [x] MLP implemented by Pytorch with Additive Encoding
- [x] CNN implemented by Pytorch with One-hot Encoding
- [x] RNN implemented by Pytorch with One-hot Encoding
- [x] Tuning models using Optuna

### Implementation - Neural Network Models for Phenotype Prediction
To implement the main code, going to the `src` folder

The program arguments include:
* `--data_dir`: directory of the source code
* `--model`: the model is used, e.g., MLP, CNN, RNN
* `--minmax`: option for normalizing y labels by min-max scaler
* `--standa`: option for normalizing X features by standardization scaler
* `--pcafit`: option for reducing the dimension of X features by PCA
* `--dataset`: dataset option, e.g., pheno1, pheno2, pheno3
* `--gpucuda`: option for training the models on GPU

Command line examples:
```bash
# Training with hyperparameter tuning
# Re-training the model with tuned hyperparameters
# Get the results on test set

python main.py --data_dir /home/ra56kop/intern_pheno_nnets/src --model MLP --minmax 0 --standa 0 --pcafit 0 --gpucuda 1 --dataset 1

```









