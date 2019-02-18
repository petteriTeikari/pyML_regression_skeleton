# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

import warnings
warnings.filterwarnings("ignore")

import math
from scipy.misc import logsumexp
import numpy as np
import sys

from keras.regularizers import l2
from keras import Input
from keras.layers import Dropout
from keras.layers import Dense, Concatenate
from keras import Model
import keras

from models.model_utils import feature_selector
from models.model_utils import split_data_to_sets
from models.model_utils import get_CV_folds
from models.metrics import compute_regression_metrics

from models.concrete_dropout import ConcreteDropout

import time

def logsumexp(a):
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max

def test(Y_true, MC_samples, D = 1):
    """
    Estimate predictive log likelihood:
    log p(y|x, D) = log int p(y|x, w) p(w|D) dw
                 ~= log int p(y|x, w) q(w) dw
                 ~= log 1/K sum p(y|x, w_k) with w_k sim q(w)
                  = LogSumExp log p(y|x, w_k) - log K
    :Y_true: a 2D array of size N x dim
    :MC_samples: a 3D array of size samples K x N x 2*D
    """
    #assert len(MC_samples.shape) == 3
    #assert len(Y_true.shape) == 2
    k = MC_samples.shape[0]
    N = Y_true.shape[0]
    mean = MC_samples[:, :, :D]  # K x N x D
    logvar = MC_samples[:, :, D:]
    test_ll = -0.5 * np.exp(-logvar) * (mean - Y_true[None])**2. - 0.5 * logvar - 0.5 * np.log(2 * np.pi)
    test_ll = np.sum(np.sum(test_ll, -1), -1)
    test_ll = logsumexp(test_ll) - np.log(k)
    pppp = test_ll / N  # per point predictive probability
    rmse = np.mean((np.mean(mean, 0) - Y_true)**2.)**0.5
    return pppp, rmse

def fit_model(nb_epoch, X, Y, X_test, y_test,
              l = 1e-4, nb_features = 58,
              D = 1, batch_size = 32, verbose = False):

    N = X.shape[0]
    wd = l ** 2. / N
    dd = 2. / N

    input_shape = X.shape[1:]
    inp = Input(input_shape)
    x = inp

    if nb_features == 'series':
        nb_list = [58,28,14, 6]
    else:
        nb_list = [nb_features, nb_features, nb_features, nb_features]

    # Dense layers
    x = ConcreteDropout(Dense(nb_list[0], activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
    x = ConcreteDropout(Dense(nb_list[1], activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
    x = ConcreteDropout(Dense(nb_list[2], activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
    # x = ConcreteDropout(Dense(nb_list[2], activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)

    mean = ConcreteDropout(Dense(D), weight_regularizer=wd, dropout_regularizer=dd)(x)
    log_var = ConcreteDropout(Dense(D), weight_regularizer=wd, dropout_regularizer=dd)(x)
    out = keras.layers.Concatenate(name = 'output')([mean, log_var])
    model = Model(inp, out)

    def heteroscedastic_loss(true, pred):
        mean = pred[:, :D]
        log_var = pred[:, D:]
        precision = keras.backend.exp(-log_var)
        return keras.backend.sum(precision * (true - mean) ** 2. + log_var, -1)

    model.compile(optimizer='adam', loss=heteroscedastic_loss)
    #assert len(model.layers[1].trainable_weights) == 3  # kernel, bias, and dropout prob
    #assert len(model.losses) == 5  # a loss for each Concrete Dropout layer
    hist = model.fit(X, Y, nb_epoch=nb_epoch, batch_size=batch_size,
                     verbose=verbose, validation_data=(X_test, y_test))
    loss = hist.history['loss'][-1]

    model.summary()

    return model, -0.5 * loss  # return ELBO up to const.


def MCdpout_grid_search(data_encoded, y_regression,
                        lscale= [1e-4],
                        nb_feat = [28],
                        nb_epochs = 2000, batch_size = 32,
                        nb_reps = 3, K_test = 20, D = 1, verbose = False):

    # https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-keras.ipynb
    # + https://github.com/yaringal/DropoutUncertaintyExps/blob/master/experiment.py

    print('\nGRID SEARCH for best hyperparameters')
    print('   ... feature count from = ', nb_feat)
    print('   ... length scale from = ',  lscale)

    # Get rid of some useless features if you want
    data_feats_kept = feature_selector(data=data_encoded,
                                       label=y_regression)

    # Make the split here
    X_train, y_train, X_test, y_test = split_data_to_sets(data=data_feats_kept,
                                                          label=y_regression,
                                                          test_ratio=0.33)

    # For CV, you could split now the train set then further to train and validation and keep the test aside
    X_train, y_train, X_validation, y_validation = get_CV_folds(X_train, y_train)

    # We perform grid-search to select the best hyperparameters based on the highest log-likelihood value
    best_network = None
    best_ll = -float('inf')
    best_tau = 0
    best_dropout = 0

    no_comb = len(lscale)*len(nb_feat)
    print('\nNumber of grid search combinations = ', no_comb)
    print('Number of epochs per iteration = ', nb_epochs, '\n')

    cpu_epoch_time = 0.0554855
    gpu_epoch_time = 0.1605

    print('   ... rough estimate for waiting with CPU (i7700) = ',
          (no_comb*nb_epochs*nb_reps*cpu_epoch_time)/60, ' minutes')
    print('   ... rough estimate for waiting withy GPU (1080Ti) = ',
          (no_comb*nb_epochs*nb_reps*gpu_epoch_time) /60, ' minutes \n')

    best_network = None
    best_metric = float('inf')
    best_lscale = 0
    best_dropouts = 0
    best_featcount = 0
    best_y_preds = 0

    for length in lscale:
        for feat_count in nb_feat:

            t0 = time.time()

            print('---------')
            print('\nGrid search step: length_scale: ' + str(length) + ' Feat count: ' + str(feat_count))
            print('---------')
            results = []

            # repeat exp multiple times
            rep_results = []
            for i in range(nb_reps):

                print('  Repeat = ', i+1, '/', nb_reps)

                model, ELBO = fit_model(nb_epochs, X_train, y_train, X_test, y_test,
                                      l = length, nb_features = feat_count, verbose = verbose,
                                        batch_size = batch_size)

                MC_samples = np.array([model.predict(X_test) for _ in range(K_test)])
                pppp, rmse = test(y_test, MC_samples)  # per point predictive probability

                means = MC_samples[:, :, :D]  # K x N
                epistemic_uncertainty = np.var(means, 0).mean(0)
                logvar = np.mean(MC_samples[:, :, D:], 0)
                aleatoric_uncertainty = np.exp(logvar).mean(0)
                ps = np.array([keras.backend.eval(layer.p) for layer in model.layers if hasattr(layer, 'p')])
                rep_results += [(rmse, ps, aleatoric_uncertainty, epistemic_uncertainty)]

                print('\trmse = ', rmse)

            test_mean = np.mean([r[0] for r in rep_results])
            test_std_err = np.std([r[0] for r in rep_results]) / np.sqrt(nb_reps)
            ps = np.mean([r[1] for r in rep_results], 0)
            aleatoric_uncertainty = np.mean([r[2] for r in rep_results])
            epistemic_uncertainty = np.mean([r[3] for r in rep_results])

            # Print results
            print('\n\tRMSE Test mean: ', test_mean)
            print('\tRMSE Test stdev: ',  test_std_err)
            print('\tDropout probabilities (per dense layer): ', ps)
            print('\tAleatoric uncertainty (exp from logvar): ', aleatoric_uncertainty ** 0.5)
            print('\tEpistemic uncertainty (var of means):', epistemic_uncertainty ** 0.5, '\n')
            sys.stdout.flush()
            results += [rep_results]

            t1 = time.time()
            print('... repeats took = ', t1 - t0, ' seconds\n')
            # ~963 seconds per repeats [1080Ti]

            metric = test_mean
            if (metric < best_metric):

                best_metric = metric
                best_network = model
                best_lscale = length
                best_featcount = feat_count
                best_dropouts = ps
                best_y_preds = np.mean(means, axis=0)
                best_aleatoric = aleatoric_uncertainty ** 0.5
                best_epistemic = epistemic_uncertainty ** 0.5
                #print(best_y_preds)
                #print(best_y_preds.shape)

                # y_best_pred = best_network.predict(X_test)
                # results = best_network.score(X_test, best_y_preds)
                compute_regression_metrics(y_test, best_y_preds,
                                           test_mean, test_std_err,
                                           best_aleatoric, best_epistemic,
                                           results, 'mcdropout_best_metrics.csv', nb_reps)

                print('Best test mean RMSE: ' + str(best_metric))
                print('Best length changed to: ' + str(best_lscale))
                print('Best feat count changed to: ' + str(best_featcount))
                print('Best dropout rate changed to: ' + str(best_dropouts))


            keras.backend.clear_session()  # Clear session after all the repeats of given combo

    return (best_metric, best_network, best_lscale, best_featcount, best_y_preds)