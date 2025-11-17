from network import initialize_network, query
from train import train_until
from store import json_save, json_load
import training_data

def main():
    # weights, biases = initialize_network(6, [7,9,7], 6)
    weights, biases = json_load('network_state.json')
    training_data.setup()
    data = training_data.DATA
    train_until(*data[9], weights, biases)
    bits = training_data.bits
    stib = training_data.stib
    
    def cycle(i):
        # user_input = input("Enter input for prediction: ")
        # print(f"user_input is equal to {user_input}")
        # Loop over each value and predict
        result = query(i, weights, biases, bits, stib)
        print(f"Predicted output for {i}: '{result}'")
        return result
    
    json_save(weights, biases, 'network_state.json')
    
    for i in training_data.DATA:
        cycle(i[0])
    

if __name__ == "__main__":
    main()
    