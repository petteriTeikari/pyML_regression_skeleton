import os
import sys
import pickle
import numpy as np

from sklearn.metrics import r2_score, explained_variance_score, median_absolute_error, \
    mean_absolute_error, mean_squared_error, mean_squared_log_error

def compute_regression_metrics(y_true, y_pred,
                               test_mean, test_std_err,
                               best_aleatoric, best_epistemic,
                               results_cv, path_out, fileout_string, nb_reps):

    # https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    # https://scikit-learn.org/stable/modules/classes.html#regression-metrics

    # cast as dtype('<U32>'), if there is a problem and your numbers are actually strings
    if y_pred.dtype.type is np.str_:
        y_pred = y_pred.astype(np.float)

    if y_true.dtype.type is np.str_:
        y_true = y_true.astype(np.float)

    r2 = r2_score(y_true, y_pred)
    exp_var = explained_variance_score(y_true, y_pred)
    mean_abs_err = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    #msle = mean_squared_log_error(y_true, y_pred)
    med_abs_err = median_absolute_error(y_true, y_pred)

    # Create the headers
    no_of_folds = nb_reps #results_cv.shape[0]
    folds = np.linspace(1, no_of_folds, no_of_folds)
    fold_str = []
    for i, item in enumerate(folds):
        fold_str.append('fold ' + str(int(item)))

    # rest of the headers
    headers2 = ['Test Mean', 'Test Stdev', 'Aleatoric Uncertainty', 'Epistemic Uncertainty', 'Explained variance', 'Mean absolute error', 'Mean Squared Error', 'Median Absolute error', 'R^2']
    headers_all = headers2 # np.hstack((headers2))
    headers_all = ",".join(headers_all)

    # stack the data
    # data_out = np.hstack((results_cv, results_cv.mean(), results_cv.std(), exp_var, mean_abs_err, mse, med_abs_err, r2))
    data_out = np.hstack((test_mean, test_std_err, best_aleatoric, best_epistemic, exp_var, mean_abs_err, mse, med_abs_err, r2))

    # and write to disk
    np.savetxt(os.path.join(path_out, fileout_string),  np.column_stack(data_out), delimiter=",", header=headers_all, fmt="%f", comments='')

    # write the predictions too
    fileout_string = fileout_string.replace('metrics.', 'predictions.')
    true_vs_pred = np.hstack((np.row_stack(y_true), np.row_stack(y_pred)))
    np.savetxt(os.path.join(path_out, fileout_string), true_vs_pred, delimiter=",", header='True,Predicted', fmt="%f", comments='')

    print('     R^2 = ', r2)
    print('     MSE = ', mse)
