import argparse

from models.RNN import *
from models.MLP import *
from models.CNN import *

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

    parser.add_argument("-ddi", "--data_dir", type=str,
                        default='/Users/nghihuynh/Documents/MscTUM_BioTech/4th_semester/Internship/intern_pheno_nnets/src',
                        help="Path to the data folder")
    
    parser.add_argument("-mod", "--model", type=str,
                        default='MLP',
                        help="NNet model for training the phenotype prediction")
    
    parser.add_argument("-min", "--minmax", type=int,
                        default=1,
                        help="Nomalizing y with min-max scaler")
    
    parser.add_argument("-sta", "--standa", type=int,
                        default=0,
                        help="Nomalizing X with min-max scaler")
    
    parser.add_argument("-pca", "--pcafit", type=int,
                        default=0,
                        help="Reducing and fitting X with PCA")
    
    parser.add_argument("-dat", "--dataset", type=int,
                        default=1,
                        help="The set of data using for training")
    
    parser.add_argument("-gpu", "--gpucuda", type=int,
                        default=0,
                        help="Training the model on GPU")

    args = vars(parser.parse_args())

    # ----------------------------------------------------
    # Check available GPUs, if not, run on CPUs
    # ----------------------------------------------------
    dev = "cpu"
    if args["gpucuda"] >= 1 and torch.cuda.is_available(): 
        print("GPU CUDA available, using GPU for training the models.")
        dev = "cuda:" + str(args["gpucuda"]-1) # to get the idx of gpu device
    else:
        print("GPU CUDA not available, using CPU instead.")
    device = torch.device(dev)

    # ----------------------------------------------------
    # Parsing the input arguments
    # ----------------------------------------------------
    datapath = args["data_dir"]
    model = args["model"]
    minmax_scale = args["minmax"]
    standa_scale = args["standa"]
    pca_fitting  = args["pcafit"]
    dataset = args["dataset"]
    gpucuda = args["gpucuda"]

    print('-----------------------------------------------')
    print('Input arguments: ')
    print('   + data_dir: {}'.format(datapath))
    print('   + model: {}'.format(model))
    print('   + minmax_scale: {}'.format(minmax_scale))
    print('   + standa_scale: {}'.format(standa_scale))
    print('   + pca_fitting: {}'.format(pca_fitting))
    print('   + dataset: pheno_{}'.format(dataset))
    print('   + gpucuda: {}'.format(gpucuda))

    data_variants = [minmax_scale, standa_scale, pca_fitting, dataset]
    print('   + data_variants: {}'.format(data_variants))
    print('-----------------------------------------------\n')

    # ----------------------------------------------------
    # Read data and preprocess
    # ----------------------------------------------------

    # Additive Encoding
    # read_data_pheno_additive(datapath, 3)
    # split_train_test_data_additive(datapath, 3)

    # One_hot Encoding
    # read_data_pheno_onehot(datapath, 1)
    # split_train_test_data_onehot(datapath, 1)

    # ----------------------------------------------------
    # Tune and evaluate the model performance
    # ----------------------------------------------------
    # set up parameters for tuning
    training_params_dict = {
        'num_trials': 100,
        'min_trials': 20,
        'percentile': 65,
        'optunaseed': 42,
        'num_epochs': 80,
        'early_stop': 20,
        'batch_size': 32
    }

    if model == 'MLP':
        print('---------------------------------------------------------')
        print('Tuning MLP with dataset pheno-{}, minmax={}, standard={}, pcafit={}'.format(dataset, minmax_scale, standa_scale, pca_fitting))
        print('---------------------------------------------------------\n')
        X_train, y_train, X_test, y_test = load_split_train_test_additive(datapath, dataset)
        best_params = tuning_MLP(datapath, X_train, y_train, data_variants, training_params_dict, device)
        evaluate_result_MLP(datapath, X_train, y_train, X_test, y_test, best_params, data_variants, device)

        
    elif model == 'CNN':
        print('---------------------------------------------------------')
        print('Tuning CNN with dataset pheno-{}'.format(dataset))
        print('---------------------------------------------------------\n')
        X_train, y_train, X_test, y_test = load_split_train_test_onehot(datapath, dataset)
        best_params = tuning_CNN(datapath, X_train, y_train, data_variants, training_params_dict, device)
        evaluate_result_CNN(datapath, X_train, y_train, X_test, y_test, best_params, data_variants, device)

    
    elif model == 'RNN':
        print('---------------------------------------------------------')
        print('Tuning RNN with dataset pheno-{}'.format(dataset))
        print('---------------------------------------------------------\n')
        X_train, y_train, X_test, y_test = load_split_train_test_onehot(datapath, dataset)
        best_params = tuning_RNN(datapath, X_train, y_train, data_variants, training_params_dict, device)
        evaluate_result_RNN(datapath, X_train, y_train, X_test, y_test, best_params, data_variants, device)

    