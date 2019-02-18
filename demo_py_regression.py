#!/usr/bin/python3
# Minimal working example on how to crunch your csv files also in Python
# Petteri Teikari, 2019

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
# from keras.datasets import boston_housing
from test_data_private.import_epidem_data import import_epi_data
from models.model_main import model_wrapper

def main():

    # Setup
    this_dir = sys.path[0]
    os.chdir(this_dir)
    path_in = os.path.join(this_dir, 'test_data_private')
    path_out = os.path.join(this_dir, 'MODELS_OUT')
    print('Output path is = ', path_out)

    # Import data
    read_mode = '' # ''non_hotencoded'
    data_encoded, y_classification, y_regression, scaler = import_epi_data(path = path_in,
                                                                           process_from_nonhotencoded = False)

    # Your model(s) here
    model_wrapper(data_encoded, y_classification, y_regression, scaler)



if __name__ == "__main__":
    main()

