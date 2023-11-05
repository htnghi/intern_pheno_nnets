import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.RNN import *
from models.MLP import *
from preprocess.encode_data import *

if __name__ == '__main__':
    """
    Run the main.py file to start the program:
        + Process the input arguments
        + Read data
        + Preprocess data
        + Train models
        + Prediction
    """

    # ----------------------------------------------------
    # Process the arguments
    # ----------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("-dd", "--data_dir", type=str,
                        default='/Users/nghihuynh/Documents/MscTUM_BioTech/4th_semester/Internship/intern_pheno_nnets/src/data',
                        help="Path to the data folder")
    
    parser.add_argument("-enc", "--encode", type=str,
                        default='012',
                        help="Mode for encoding the datasets")
    
    parser.add_argument("-mod", "--model", type=str,
                        default='RNN',
                        help="NNet model for training the phenotype prediction")

    args = vars(parser.parse_args())

    # ----------------------------------------------------
    # Read data and preprocess
    # ----------------------------------------------------
    datapath = args["data_dir"]
    # print("Data dir: ", datapath)
    # print('-----------------------------------------------\n')

    read_data(datapath)
    X, y = read_prerocessed_data(datapath)
    # print("Data after preprocessing: ")
    # print(X)
    # print(y)
    # print('-----------------------------------------------\n')

    # ----------------------------------------------------
    # Encode data
    # ----------------------------------------------------
    # X_additive_encoded = get_additive_encoding(X)
    # print("Data after additive encoding: ")
    # print(X_additive_encoded)
    # print('-----------------------------------------------\n')

    X_onehot_encoded = get_onehot_encoding(X)
    # print("Data after one-hot encoding: ")
    # print(X_onehot_encoded)
    # print('-----------------------------------------------\n')

    #y_onehot_encoded = y_onehot(y)
    # print("Data after one-hot encoding: ")
    # print(y_onehot_encoded)
    # print('-----------------------------------------------\n')

    # ----------------------------------------------------
    # Train model
    # ----------------------------------------------------
    model = run_train_RNN(X_onehot_encoded, y)

    #model = run_train_RNN(X_onehot_encoded, y_onehot_encoded)
    
