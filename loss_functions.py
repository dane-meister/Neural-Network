import numpy as np


def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length
    return ((y_true - y_pred) ** 2).mean()


def mse_loss_prime(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length
    return -2 * (y_true - y_pred)


def cross_entropy_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length
    return -np.sum(y_true * np.log(y_pred))


def cross_entropy_loss_prime(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length
    return -y_true / y_pred


def binary_cross_entropy_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_loss_prime(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length
    return -y_true / y_pred + (1 - y_true) / (1 - y_pred)


