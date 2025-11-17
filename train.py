import numpy as np
from activations import sigmoid, sigmoid_derivative
from forward import forward_pass
from network import predict
import training_data



def train_network(td, w, b, lr=0.1, e=1000):
    for epoch in range(e):
        total_error = 0
        for input_val, expected_output in td:
            total_error += train_once(input_val, expected_output, lr, w, b)
        if epoch % 1000 ==0:
            print(f"Epoch {epoch}, Total Error: {total_error}")

    return w, b


def train_once(input_val, expected_output, learning_rate, weights, biases):
    """
    # Example of adjusting training data to include negative example flag
    training_data = [
        # Positive examples: ((input_features), (target_output, False))
        ((input_features_for_example_1), (target_output_for_example_1, False)),
        # Negative examples: ((input_features), (target_output, True))
        ((input_features_for_negative_example_1), (target_output_for_negative_example_1, True)),
    ]

    """
    # Forward pass: Store activations for each layer
    activations = forward_pass(input_val, weights, biases)

    # Backward pass
    # Calculate output error
    output_error = expected_output - activations[-1]
    total_error = np.sum(output_error ** 2)

    # Calculate gradient for output layer
    d_error = output_error * sigmoid_derivative(activations[-1])

    for i in reversed(range(len(weights))):
        # Calculate error for the current layer
        d_weights = np.outer(activations[i], d_error)
        d_biases = d_error

        # Update weights and biases
        weights[i] += learning_rate * d_weights
        # cannot be: `biases[i] += learning_rate * d_biases` due to numpy
        biases[i] = biases[i] + (learning_rate * d_biases)

        # Propagate the error backward
        if i > 0:
            d_error = np.dot(d_error, weights[i].T) * sigmoid_derivative(activations[i])
    return total_error


def train_until(test_word, test_expected, weights, biases):
    num_data = training_data.convert_data_to_floats(training_data.DATA)
    result = None
    count = 0
    success = False

    while success is False or count <= 10:
        count += 1
        train_network(num_data, weights, biases)
        result = predict(training_data.get_word(test_word), weights, biases, training_data.bits, training_data.stib)
        success = result == test_expected
        print(f"'{result}'")
        if success is True: break
        