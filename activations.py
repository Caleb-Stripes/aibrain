"""
Activation functions and their derivatives for use in machine learning models.
Includes sigmoid and its variants for neural network computations.
"""
import numpy as np


def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))


def sigmoid_derivative(x):
    return x * (1 - x)
