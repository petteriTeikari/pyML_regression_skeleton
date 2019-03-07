
import numpy as np
from models.MC_dropout_dense_regression import MCdpout_grid_search

import time

def model_wrapper(data_encoded, y_classification, y_regression, scaler, path_out,
                  verbose = True,
                  model = 'MC_dropout', do_grid_search = True):

    if verbose:
        print('\nInput training data dimension:', data_encoded.shape)
        print('   continous regression target vector length = ', len(y_classification))
        print('     discrete classification label vector length = ', len(y_regression))

        # We have done z=standardization for the input and we saved the mean and std used for that
        print('        scaler object saved when preprocessing data = "', scaler, '"')
        print('        i.e. if you want the data scaled back to input range, or use the model to do predictions on new data\n')

    # Throw models here

    # Test the MC dropout model from Yarin Gal et al.
    # https://github.com/yaringal/DropoutUncertaintyExps/blob/master/experiment.py
    print(model)
    if model == 'MC_dropout':

        print('Model = ', model)
        if do_grid_search:

            print('Grid Search for optimal hyperparameters')
            nb_feat = [data_encoded.shape[1]]
            best_metric, best_network, best_lscale, best_featcount, best_y_preds = MCdpout_grid_search(data_encoded, y_regression, path_out,
                                                                           nb_feat =  nb_feat)

            print('test mean RMSE = ', best_metric, ', lscale = ',  best_lscale, ', feat_count = ', best_featcount)
            # TODO! Save these to disk as well

        else:

            print('Skipping hyperparameters and using optimized values')

    else:

        print('Used model = "', model, '" not supported')