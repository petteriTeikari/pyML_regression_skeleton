import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler

def import_epi_data(path='/home/petteri/Dropbox/manuscriptDrafts/deepPLR/code/RLN_tabularData/test_data_private',
                    test_dataset='TestPredMod.csv', process_from_nonhotencoded = True,
                    # file_in = 'df_cont_ZCA_cor.csv',
                    # file_in = 'df_scaled.csv',
                    file_in = 'df_ZCA_cor.csv',
                    debug_mode=False, verbose = True):

    import csv

    if process_from_nonhotencoded:

        with open(os.path.join(path, test_dataset), "rt") as f:
            reader = csv.reader(f)
            headers = np.asarray(next(reader))
            data = [row for row in reader]
            data = np.asarray(data)

        shape_in = data.shape
        no_rows = shape_in[0]
        no_cols = shape_in[1]

        binary_label_col = 2 # remember, Python starts from zero
        continuous_label_col = 3  # remember, Python starts from zero
        data_indices = np.repeat(True, no_cols)
        data_indices[binary_label_col] = False
        data_indices[continuous_label_col] = False
        y_classification = data[:,binary_label_col].astype(np.float)
        y_regression = data[:,continuous_label_col].astype(np.float)

        headers_wo_labels = headers[data_indices]
        data_wo_labels = data[:,data_indices].astype(np.float)

        categorical_label_names = ['gender', 'any_retino', 'smk_cat', 'alc_cat', 'edu_cat', 'job_cat2', 'dm5',
                                   'hypertension', 'anti_ht', 'anti_chol', 'anti_dm']

        categ_indices_boolean = np.repeat(False, data_wo_labels.shape[1])
        for idx, categ_label in enumerate(categorical_label_names):
            match = np.where([categ_label in i for i in headers_wo_labels])[0]
            categ_indices_boolean[match] = True

        linear_indices = np.where(categ_indices_boolean)[0]

        if verbose:
            print(' We found ', len(linear_indices), ' categorical features')
            print('   and continuous ', no_cols - len(linear_indices) - 2, ' continuous indices')

        # One-hot encode the categorical variables, should be compatible with most of the algorithms?
        categ_data = data_wo_labels[:,linear_indices]
        enc = OneHotEncoder(sparse = False)
        categ_encoded = enc.fit_transform(categ_data)
        if verbose:
            print('Categorical data | from : ', categ_data.shape, ' to ', categ_encoded.shape)

        # One-hot headers correctly sorted as well:
        # https://stackoverflow.com/questions/49433462/python-sklearn-how-to-get-feature-names-after-onehotencoder
        # https://github.com/sebp/scikit-survival
        # Tutorial: https://nbviewer.jupyter.org/github/sebp/scikit-survival/blob/master/examples/00-introduction.ipynb
        categ_names = []
        for idx, runner in enumerate(categorical_label_names):
            categ_data_on_header = categ_data[:,idx]
            unique_labels = np.unique(categ_data_on_header)
            for idx, label in enumerate(unique_labels):
                categ_names.append(runner + '_' + str(label))
        # Todo! You could have a lookup table giving you human readable label names

        # Just extract the continuous variables
        contin_data = data_wo_labels[:, np.logical_not(categ_indices_boolean)]
        contin_headers = headers_wo_labels[np.logical_not(categ_indices_boolean)]
        if verbose:
            print('Continuous data: ', contin_data.shape)

        # Scale features, i.e. z-standardize
        scaler = StandardScaler()
        contin_data_z = scaler.fit_transform(contin_data) #, y_regression)

        # concatenate
        data_encoded = np.hstack((contin_data_z, categ_encoded))
        data_encoded_unscaled = np.hstack((contin_data, categ_encoded))

        # Main header
        header_main = np.hstack((contin_headers, categ_names))
        header_main = np.hstack((header_main, 'class label', 'regression label'))  # TODO! Hard-coded

        # data type header
        continuous_subheaders = np.tile(np.array('continuous'), contin_data.shape[1])
        categorical_subheaders = np.tile(np.array('categorical'), categ_encoded.shape[1])
        subheaders = np.hstack((continuous_subheaders, categorical_subheaders))
        subheaders = np.hstack((subheaders, 'classification', 'regression')) # TODO! Hard-coded

        # So as SUMMARY, you have your data in these variables
        # if verbose:
        #     print(header_main)
        #     print(subheaders)
        #     print(contin_data_z)
        #     print(categ_encoded)

        INPUT_DIM = data_encoded.shape[1]
        if verbose:
            print('Number of input features = ', INPUT_DIM)
            print('Train set size = ', data_encoded.shape, ', labels = ', y_regression.shape)
            print('\nSaving all this to disk back')
            print('    main header length = ', len(header_main))
            print('    subheader length = ', len(subheaders))

        # Labels have size (1906,) which Python distinguishes from (1906,1)
        # so the extra dimension is the "1Dification" of the labels
        df_scaled = pd.DataFrame(data = np.hstack((data_encoded,
                                                   y_classification[:, np.newaxis],
                                                   y_regression[:, np.newaxis])),
                                 columns=[header_main, subheaders])

        df_unscaled = pd.DataFrame(data = np.hstack((data_encoded_unscaled,
                                                     y_classification[:, np.newaxis],
                                                     y_regression[:, np.newaxis])),
                                   columns=[header_main, subheaders])

        # Save to disk (easier to continue in R for example if otherwise you do not like Python)
        df_unscaled.to_csv(os.path.join(path, 'df_unscaled.csv'), index=False)
        df_scaled.to_csv(os.path.join(path, 'df_scaled.csv'), index=False)

    else:

        print('\nLoad data from: ', file_in, '\n')

        with open(os.path.join(path, file_in), "rt") as f:

            reader = csv.reader(f)
            headers = np.asarray(next(reader))
            data = [row for row in reader]
            data = np.asarray(data)

            print(data.shape)
            number_of_cols = data.shape[1]

            data_encoded = data[:,0:number_of_cols-2].astype(np.float)
            y_classification = data[:,number_of_cols-2].astype(np.int)
            y_regression = data[:, number_of_cols-1].astype(np.float)
            scaler = []

            print('Train set size = ', data_encoded.shape, ', labels = ', y_regression.shape)
            # print(data_encoded[0:111,27])

    print('Import done!')
    return (data_encoded, y_classification, y_regression, scaler)