import argparse
import json

from models.RNN import *
from models.MLP import *
from models.CNN import *

from tuning.trial_tuning_MLP import *
from tuning.trial_tuning_CNN import *
from tuning.trial_tuning_RNN import *

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
    
    parser.add_argument("-tun", "--tuned", type=int,
                        default=0,
                        help="Tuning or final training the model")
    
    parser.add_argument("-tfi", "--tuned_file", type=str,
                        default='./tuning/tuned_parameters',
                        help="Path to the file with tuned parameters")
    
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
    tuned = args["tuned"]
    minmax_scale = args["minmax"]
    standa_scale = args["standa"]
    pca_fitting  = args["pcafit"]
    dataset = args["dataset"]
    gpucuda = args["gpucuda"]

    print('-----------------------------------------------')
    print('Input arguments: ')
    print('   + data_dir: {}'.format(datapath))
    print('   + model: {}'.format(model))
    print('   + tuned: {}'.format(tuned))
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
    # Trial train and tune model
    # ----------------------------------------------------
    if tuned == 1:
        # set up parameters for tuning
        training_params_dict = {
            'num_trials': 20,
            'min_trials': 20,
            'percentile': 65,
            'optunaseed': 42,
            'num_epochs': 50,
            'early_stop': 20,
            'batch_size': 32
        }

        if model == 'MLP':
            print('---------------------------------------------------------')
            print('Tuning MLP with dataset pheno-{}, minmax={}, standard={}, pcafit={}'.format(dataset, minmax_scale, standa_scale, pca_fitting))
            print('---------------------------------------------------------\n')
            X_train, y_train, X_test, y_test = load_split_train_test_additive(datapath, dataset)
            model = tuning_MLP(datapath, X_train, y_train, data_variants, training_params_dict, device)

        elif model == 'CNN':
            print('---------------------------------------------------------')
            print('Tuning CNN with dataset pheno-{}'.format(dataset))
            print('---------------------------------------------------------\n')
            X_train, y_train, X_test, y_test = load_split_train_test_onehot(datapath, dataset)
            model = tuning_CNN(datapath, X_train, y_train, data_variants, training_params_dict, device)
        
        elif model == 'RNN':
            print('---------------------------------------------------------')
            print('Tuning RNN with dataset pheno-{}'.format(dataset))
            print('---------------------------------------------------------\n')
            X_train, y_train, X_test, y_test = load_split_train_test_onehot(datapath, dataset)
            model = tuning_RNN(datapath, X_train, y_train, data_variants, training_params_dict, device)
    else:
        # ----------------------------------------------------
        # Load tuned information
        # ----------------------------------------------------
        tuned_filepath = args["tuned_file"]
        tuned_content = open(tuned_filepath, 'r').readline()

        # testing the file
        # python main.py --data_dir /home/ctminh/projects/nnets_genomic_prediction/src --model MLP --tuned 0 \
        # --tuned_file ./tuning/tuned_parameters/mlp/mlp_expvar_data1_minmax_fixedoutfactor.json --minmax 1 --standa 0 --pcafit 0 --dataset 1

        # python main.py --data_dir /home/ctminh/projects/nnets_genomic_prediction/src --model CNN --tuned 0 \
        # --tuned_file ./tuning/tuned_parameters/cnn/cnn_expvar_data1_minmax.json --minmax 1 --standa 0 --pcafit 0 --dataset 1

        # python main.py --data_dir /home/ctminh/projects/nnets_genomic_prediction/src --model RNN --tuned 0 \
        # --tuned_file ./tuning/tuned_parameters/rnn/rnn_expvar_data1_minmax.json --minmax 1 --standa 0 --pcafit 0 --dataset 1

        filename = tuned_filepath.split('/')[-1]
        filename_tokens = filename.split('_')
        dbcheck_model = filename_tokens[0]
        dbcheck_obj_direction = filename_tokens[1]
        dbcheck_dataset = filename_tokens[2]
        dbcheck_yscale_minmax = filename_tokens[3]
        dbcheck_xscale_standa = ''
        dbcheck_xscale_pcafit = ''
        print('DB_CHECK: tuned_file hyperparameters {}'.format(filename))
        print('  + dbcheck_model: {}'.format(dbcheck_model))
        print('  + dbcheck_obj_direction: {}'.format(dbcheck_obj_direction))
        print('  + dbcheck_dataset: {}'.format(dbcheck_dataset))
        print('  + dbcheck_yscale_minmax: {}'.format(dbcheck_yscale_minmax))
        print('----------------------------------------------------------\n')

        # extract the details
        tuned_content_jsonformat = json.loads(tuned_content.replace("'", "\""))
        hyperparameters = tuned_content_jsonformat

        # ----------------------------------------------------
        # Final train the model
        # ----------------------------------------------------
        if model == 'MLP':
            print('---------------------------------------------------------')
            print('Final train MLP with dataset pheno-{}, minmax={}, standard={}, pcafit={}'.format(dataset, minmax_scale, standa_scale, pca_fitting))
            print('---------------------------------------------------------\n')
            X_train, y_train, X_test, y_test = load_split_train_test_additive(datapath, dataset)
            model = run_train_MLP(datapath, X_train, y_train, X_test, y_test, hyperparameters, data_variants)
        elif model == 'CNN':
            print('---------------------------------------------------------')
            print('Final train CNN with dataset pheno-{}, minmax={}, standard={}, pcafit={}'.format(dataset, minmax_scale, standa_scale, pca_fitting))
            print('---------------------------------------------------------\n')
            X_train, y_train, X_test, y_test = load_split_train_test_onehot(datapath, dataset)
            model = run_train_CNN(datapath, X_train, y_train, X_test, y_test, hyperparameters, data_variants)
        else:
            print('---------------------------------------------------------')
            print('Final train RNN with dataset pheno-{}, minmax={}, standard={}, pcafit={}'.format(dataset, minmax_scale, standa_scale, pca_fitting))
            print('---------------------------------------------------------\n')
            X_train, y_train, X_test, y_test = load_split_train_test_onehot(datapath, dataset)
            model = run_train_RNN(datapath, X_train, y_train, X_test, y_test, hyperparameters, data_variants, device)
    