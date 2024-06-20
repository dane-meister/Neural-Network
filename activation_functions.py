import numpy as np

def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def relu(x):
    return np.maximum(0, x)


def deriv_relu(x):
    return np.where(x <= 0, 0, 1)


def tanh(x):
    return np.tanh(x)


def deriv_tanh(x):
    return 1 - np.tanh(x) ** 2


def softmax(x):
    # Numerically stable softmax
    exps = np.exp(x - x.max())
    return exps / exps.sum()


def deriv_softmax(x):
    # Derivative of softmax is softmax * (1 - softmax)
    return softmax(x) * (1 - softmax(x))

