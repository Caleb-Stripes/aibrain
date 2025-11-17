import numpy as np
from activations import sigmoid


def forward_pass_one(input_data, weights, biases):
    return forward_pass(input_data, weights, biases)[-1]


def forward_pass(input_data, weights, biases):
    activations = [input_data]
    for i in range(len(weights)):
        # on error: likely due to the input shape not matching the input nodes.
        input_data = np.dot(input_data, weights[i]) + biases[i]
        input_data = sigmoid(input_data)
        activations.append(input_data)
    return activations
