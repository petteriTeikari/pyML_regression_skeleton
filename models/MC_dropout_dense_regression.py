# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

import warnings
warnings.filterwarnings("ignore")

import math
from scipy.misc import logsumexp
import numpy as np
import sys
import os

from keras.regularizers import l2
from keras import Input
from keras.layers import Dropout
from keras.layers import Dense, Concatenate, Multiply
from keras import Model
import keras
from keras.callbacks import Callback


import pandas as pd

from models.model_utils import feature_selector
from models.model_utils import split_data_to_sets
from models.model_utils import get_CV_folds
from models.metrics import compute_regression_metrics

from models.concrete_dropout import ConcreteDropout
from models.attn_utils import attn_block

import time

class TrackConcreteDropoutP(Callback):

    # check the p values of the concrete dropout are converging.
    # https://github.com/jenny-nlc/adversarial/blob/28805a70a938cb75b592e10c161f2b597474f47a/train_cdropout_model.py

    # TODO! this only gets the p's from the layers that have the concrete_dropout directly visible
    # i.e. the dense layers, whereas the convolutional p's are hidden in the submodule.

    def __init__(self, model: Model):
        self.model = model

    def on_train_begin(self, logs={}):
        self.ps = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):

        from keras.backend import function

        self.losses.append(logs.get('loss'))
        ps_tensor = [x.p for x in self.model.layers if 'concrete_dropout' in x.name]
        # TODO! get the p inside the "ConvolutionalLayers" submodel as well
        get_ps = function([], ps_tensor)
        p = get_ps([])
        self.ps.append(p)
        print(" - concrete dropout probs (p): ", p)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

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

def dense_layer(input, nb_feat, n, prev_input, wd, dd, activ = 'relu',
                skip_conn = True, use_attn = True, # use_BN = True,
                attn_config = 'oninput'):

    output = ConcreteDropout(Dense(nb_feat, activation=activ), weight_regularizer=wd, dropout_regularizer=dd)(input)
    output_final = output

    output = keras.layers.normalization.BatchNormalization()(output)
    shortcut_y = keras.layers.normalization.BatchNormalization()(input)
    output_final = output

    if skip_conn:
        output_w_skip = keras.layers.add([shortcut_y, output_final])
        output_w_skip = keras.layers.Activation('relu')(output_w_skip)
        output_final = output_w_skip

    if use_attn:  # ATTENTION for SKIP CONNECTION

        # For inspiration see, Figure 2 of Oktay et al. (2018)
        # https://openreview.net/pdf?id=Skft7cijM
        if attn_config == 'oninput':
            if n == 0:
                attention_probs = Dense(nb_feat, activation='sigmoid', name='attention_vec_%s' % (n))(output_final)
                attention_mul = Multiply(name='attention_mul_%s' % (n))([input, attention_probs])
                output_gated = Dense(nb_feat, activation='sigmoid')(attention_mul)
                output_final = output_gated
        elif attn_config == 'onall':
            attention_probs = Dense(nb_feat, activation='sigmoid', name='attention_vec_%s' % (n))(output_final)
            attention_mul = Multiply(name='attention_mul_%s' % (n))([input, attention_probs])
            output_gated = Dense(nb_feat, activation='sigmoid')(attention_mul)
            output_final = output_gated

    return(output_final)

def fit_model(nb_epoch, X, Y, X_test, y_test, path_out,
              l = 1e-4, nb_features = 58, nb_dense_layers = 3,
              skip_conn = True, use_attn = True, attn_config = 'oninput',
              D = 1, batch_size = 32, verbose = False):

    N = X.shape[0]
    wd = l ** 2. / N
    dd = 2. / N

    input_shape = X.shape[1:]
    inp = Input(input_shape)
    x = inp

    if nb_features == 'series':
        nb_list = [58,28,14,6]
    else:
        nb_list = [nb_features, nb_features, nb_features, nb_features]
        fixed_nb_feat = nb_features

    path_out = path_out + '_attncfg-' + attn_config
    if os.path.exists(path_out) == False:
        os.makedirs(path_out)

    # Dense layers
    prev_input = x
    for n in range(nb_dense_layers):
        x2 = dense_layer(x, nb_features, n, prev_input,
                        wd, dd, skip_conn = skip_conn, use_attn = use_attn, activ = 'relu',
                        attn_config = attn_config)
        prev_input = x # for extra-long skip connection
        x = x2

    mean = ConcreteDropout(Dense(D), weight_regularizer=wd, dropout_regularizer=dd)(x)
    log_var = ConcreteDropout(Dense(D), weight_regularizer=wd, dropout_regularizer=dd)(x)
    out = keras.layers.Concatenate(name = 'output')([mean, log_var])
    model = Model(inp, out)

    model.summary()

    def heteroscedastic_loss(true, pred):
        mean = pred[:, :D]
        log_var = pred[:, D:]
        precision = keras.backend.exp(-log_var)
        return keras.backend.sum(precision * (true - mean) ** 2. + log_var, -1)

    # tensorboard = keras.callbacks.TensorBoard(
    #     log_dir='/home/petteri/Dropbox/manuscriptDrafts/deepPLR/code/data_for_deepLearning/tf_logs',
    #     histogram_freq=100, write_grads=True,
    #     embeddings_freq=0, write_graph=False,
    #     embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
    #     update_freq='epoch')  # https://github.com/keras-team/keras/issues/6674

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=20, min_lr=0.00005)
    file_path = os.path.join(path_out, 'best_model.hdf5')
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
    callbacks = [TrackConcreteDropoutP(model), reduce_lr, model_checkpoint] #, tensorboard]

    model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                  loss=heteroscedastic_loss)
    #assert len(model.layers[1].trainable_weights) == 3  # kernel, bias, and dropout prob
    #assert len(model.losses) == 5  # a loss for each Concrete Dropout layer
    hist = model.fit(X, Y, nb_epoch=nb_epoch, batch_size=batch_size,
                     verbose=verbose, validation_data=(X_test, y_test), callbacks=callbacks)
    loss = hist.history['loss'][-1]


    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(os.path.join(path_out, 'history.csv'), index=False)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.semilogy(hist.history['loss'])
    plt.semilogy(hist.history['val_' + 'loss'])
    plt.title('MODEL TRAINING: ' + 'log loss')
    plt.ylabel('heteroscedastic loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(os.path.join(path_out, 'loss.png'))
    plt.close()



    return model, -0.5 * loss, path_out  # return ELBO up to const.


def MCdpout_grid_search(data_encoded, y_regression, path_out,
                        lscale= [1,1e-1,1e-2], # [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                        nb_feat = [58], no_layers = [2],
                        nb_epochs = 1000, batch_size = 32,
                        nb_reps = 1, K_test = 100, D = 1, verbose = True):

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

    path_out_base = path_out

    for length in lscale:
        for feat_count in nb_feat:
            for no_lay in no_layers:

                t0 = time.time()

                print('---------')
                print('\nGrid search step: length_scale: ' + str(length) + ' Feat count: ' + str(feat_count) + ' No of layers: ' + str(feat_count))
                print('---------')
                results = []

                # repeat exp multiple times
                rep_results = []
                for i in range(nb_reps):

                    print('  Repeat = ', i+1, '/', nb_reps)
                    path_out = os.path.join(path_out_base, ('l-' + str(length) + '_f-' + str(feat_count) + '_n-' + str(no_lay) + '_rep-' + str(i)))

                    model, ELBO, path_out = fit_model(nb_epochs, X_train, y_train, X_test, y_test, path_out,
                                          l = length, nb_features = feat_count,
                                            nb_dense_layers = no_lay,
                                            verbose = verbose, batch_size = batch_size)

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
                                               results,
                                               path_out, 'mcdropout_best_metrics.csv', nb_reps)

                    print('Best test mean RMSE: ' + str(best_metric))
                    print('Best length changed to: ' + str(best_lscale))
                    print('Best feat count changed to: ' + str(best_featcount))
                    print('Best dropout rate changed to: ' + str(best_dropouts))


                keras.backend.clear_session()  # Clear session after all the repeats of given combo

    return (best_metric, best_network, best_lscale, best_featcount, best_y_preds)