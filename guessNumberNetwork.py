# Initialization of a NeuralNetwork istance

from NeuralNetwork import *

# number of input, hidden and output nodes
input_nodes = 28 * 28
hidden_nodes = 200
output_nodes = 10
# learning rate is 0.2
learning_rate = 0.2
# create instance of neural network
ngn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

if ngn.importWeights("weights.txt"):
    print("Pesi importati correttamente.")
else:
    print("Nessun peso importato.")