import numpy as np
from activation_functions import sigmoid

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    

    def feedforward(self, inputs):
        # Weight inputs, add bias, use activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

