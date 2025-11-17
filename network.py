import numpy as np
from activations import sigmoid
from forward import forward_pass_one, forward_pass
import training_data


def initialize_network(input_size: int, hidden_layers: list, output_size: int):
    layers = [input_size] + hidden_layers + [output_size]
    weights = []
    biases = []
    print(f' initialize_network {layers=}')
    
    l = len(layers) - 1

    for i in range(l):
        
        weight = np.random.uniform(-1, 1, (layers[i], layers[i+1]))
        bias = np.random.uniform(-1, 1, (layers[i+1],))
        print(f' #{i+1}/{l} {weight=}')
        print(f'   {bias=}\n')
        weights.append(weight)
        biases.append(bias)

    return weights, biases


def predict(input_val, weights, biases, bits, stib):
    input_val = np.array(input_val)
    v = forward_pass_one(input_val, weights, biases)
    res = ''
    for final_output in v:
        closest_num = min(bits.values(), key=lambda x: abs(x - final_output))
        predicted_letter = stib[closest_num]
        res += predicted_letter
    return res


def query(value: str, weights, biases, bits, stib):
    value = value.ljust(6)
    result = predict(training_data.get_word(value), weights, biases, bits, stib)
    return result
