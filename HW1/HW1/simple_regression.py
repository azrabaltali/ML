## Simple Regression
#     Make sure you add the bias feature to each training and test example.
#     Standardize the features using the mean and std computed over training data.

import sys
import numpy as np
from matplotlib import pyplot as plt
import scaling


# Read data matrix X and labels y from text file.
def read_data(file_name):
#  YOUR CODE HERE
  X = []
  y = []

  return X, y


# Implement gradient descent algorithm to compute w = [w0, w1].
def train(X, y, lamda, epochs):
#  YOUR CODE HERE
  w = np.zeros(w.shape)

  return w


# Compute Root mean squared error (RMSE)).
def compute_rmse(X, y, w):
#  YOUR CODE HERE
  return 0


# Compute objective (cost) function.
def compute_cost(X, y, w):
#  YOUR CODE HERE
  return 0


# Compute gradient descent Algorithm.
def compute_gradient(X, y, w):
#  YOUR CODE HERE
  grad = np.zeros(w.shape)

  return grad



##======================= Main program =======================##

# Read the training and test data.
Xtrain, ttrain = read_data("train.txt")
Xtest, ttest = read_data("test.txt")

#  YOUR CODE HERE

