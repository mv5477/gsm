import os

train_data_folder = os.path.abspath(os.path.join('data', 'results'))
test_data_folder = os.path.abspath(os.path.join('data', 'results_test'))

# model parameters
N_DIMS = 32
ALPHA = 0.4
BETA_ONE = 0.1
BETA_TWO = 0.3
EPSILON = 0.01
K_NN = 5
GAMMA = 0.5
