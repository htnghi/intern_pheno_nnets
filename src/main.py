import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.RNN import *
from models.MLP import *
from models.CNN import *
from tuning.trial_tuning_CNN import *
from tuning.trial_tuning_MLP import *
from preprocess.encode_data import *

if __name__ == '__main__':

    # ----------------------------------------------------
    # Process the arguments
    # ----------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("-dd", "--data_dir", type=str,
                        default='/Users/nghihuynh/Documents/MscTUM_BioTech/4th_semester/Internship/intern_pheno_nnets/src',
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

    # Additive Encoding
    # read_data_pheno_additive(datapath, 3)
    # split_train_test_data_additive(datapath, 3)
    # X1_train, y1_train, X1_test, y1_test = load_split_train_test_additive(datapath, 1)

    # One_hot Encoding
    # read_data_pheno_onehot(datapath, 1)
    # split_train_test_data_onehot(datapath, 1)
    X1_train, y1_train, X1_test, y1_test = load_split_train_test_onehot(datapath, 1)

    # ----------------------------------------------------
    # Train model
    # ----------------------------------------------------

    # model = run_train_CNN(datapath, X1_train, y1_train, X1_test, y1_test)

    # model = run_train_MLP(datapath, X1_train, y1_train, X1_test, y1_test)

    # model = trial_train_and_tune_CNN(datapath, X1_train, y1_train)
    
    # model = trial_train_and_tune_MLP(datapath, X1_train, y1_train)

    model = run_train_RNN(datapath, X1_train, y1_train, X1_test, y1_test)



