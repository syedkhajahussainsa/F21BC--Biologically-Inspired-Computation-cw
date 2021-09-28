import numpy as np


class MLP(object):

    def __init__(self, num_inputs=8, hidden_layers=[8,20,10 ], num_outputs=1):

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights


    def forward_propagate(self, inputs):

        # the input layer activation is just the input itself
        activations = inputs

        # iterate through the network layers
        for w in self.weights:

            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

        # return output layer activation
        return activations


    def _sigmoid(self, x):
        
        y = 1.0 / (1 + np.exp(-x))
        return y


if __name__ == "__main__":

    # create a Multilayer Perceptron
    mlp = MLP()

    # set random values for network's input
    inputs = np.random.rand(mlp.num_inputs)

    # perform forward propagation
    output = mlp.forward_propagate(inputs)
    print("Network activation inout: {}".format(inputs))

    print("Network activation: {}".format(output))
