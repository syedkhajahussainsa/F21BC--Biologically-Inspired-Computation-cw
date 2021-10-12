import numpy as np

class MLP():

    def __init__(self, numinputs=8, hiddenlayers=[8,20,10 ], numoutputs=1):

        self.numinputs = numinputs
        self.hiddenlayers = hiddenlayers
        self.numoutputs = numoutputs

        # create a generic representation of the layers
        layers = [numinputs] + hiddenlayers + [numoutputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights


    def forward_propagate( inputs, weights):
        ## bias?
        # the input layer activation is just the input itself
        activations = inputs
        net_inputs = []
        z_value = []
        activated_value = []
        activated_value.append(np.array(df_X.iloc[1,:]))
        # iterate through the network layers
        for w in weights:
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = (np.dot(activations, w))
            vfunc = np.vectorize(_sigmoid)

            # apply sigmoid activation function
            activations = vfunc(net_inputs)

            z_value.append(net_inputs)
            activated_value.append(activations)
       
        # return output layer activation
        return z_value, weights, activated_value
    

     def back_propogate( z_value, weights, activated_value ): 
        
            eval = (1/(len(activated_value[-1]))) * sum(y-activated_value[-1])
            vfunc = np.vectorize(_sigmoid_derv)
            sig_derv_activ = [vfunc(x) for x in z_value]
            weight_update = []

            for i in range(len(weights)-1, -1,-1):
                    sig_derv_activ[i] = sig_derv_activ[i] * eval
                    weight_update.insert(0,np.dot(np.reshape(sig_derv_activ[i], (-1, 1)),np.reshape(activated_value[i],(-1,1)).transpose()).transpose())
                    #add alpha

                    eval = np.dot(np.reshape(sig_derv_activ[i], (-1, 1)).transpose(),weights[i].transpose())


    
             return  [ weights[i] - weight_update[i] for i in range(len(weights))]
    
    def _sigmoid(self, x):
        
        return 1.0 / (1 + np.exp(-x))
        
    
    def _relu(self, x): 
        return max(0, x)
    
    def _tanh(self, x): 
         return (np.exp(x) - np.exp(-x))/ (np.exp(x)+ np.exp(-x))
        
        
    def _sigmoid_derv(self, x, e): 
        val = _sigmoid(x) 
        return (val - (1-val)) * e
       

    def _relu_derv(self, x): 
        return 1 if x>0 else 0
    
    def _tanh_derv(self, x):
        return 1 - (((np.exp(x) - np.exp(-x))**2)/ ((np.exp(x)+ np.exp(-x))**2))

    

        

if __name__ == "__main__":

    # create a Multilayer Perceptron
    mlp = MLP()

    # set random values for network's input
    inputs = np.random.rand(mlp.numinputs)

    # perform forward propagation
    output = mlp.forward_propagate(inputs)
    print("Network activation input: {}".format(inputs))

    print("Network activation: {}".format(output))
