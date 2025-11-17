import json
import numpy as np

def json_save(weights, biases, filename):
    data = {
        'weights': [w.tolist() for w in weights],
        'biases': [b.tolist() for b in biases]
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
        
        
def json_load(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    weights = [np.array(w) for w in data['weights']]
    biases = [np.array(b) for b in data['biases']]
    return weights, biases