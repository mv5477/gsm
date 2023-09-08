import sys
import os

train_data_folder = os.path.abspath(os.path.join('data', 'results'))
test_data_folder = os.path.abspath(os.path.join('data', 'results_test'))

# model parameters
n_dims = 32
alpha = 0.4
beta_one = 0.1
beta_two = 0.3
epsilon = 0.01
k_nn = 5
gamma = 0.5