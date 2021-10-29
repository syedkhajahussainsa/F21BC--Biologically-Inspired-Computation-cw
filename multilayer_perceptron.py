import numpy as np
#import sys
class MLP():

    def __init__(self, numinputs=4, hiddenlayers=[4,2,2,1], numoutputs=1, activationfunc = [1,1,1], alpha = 0.25, lossfunc = 1):
        self.numinputs = numinputs
        self.hiddenlayers = hiddenlayers
        self.numoutputs = numoutputs

        # create a generic representation of the layers
        layers = [numinputs] + hiddenlayers + [numoutputs]
         
        # create random connection weights for the layers
        weights = []
        bias = []
        for i in range(len(layers)-1):
            bias.append(np.ones(layers[i+1]))
            weights.append(np.random.rand(layers[i], layers[i+1]))
        print(layers)
        self.weights = weights
        self.bias = bias
        self.alpha = alpha
        
        
        if lossfunc == 1: 
            self.lss = self.loss_func
        elif lossfunc == 2: 
            self.lss = self.cross_entropy_loss
    
        self.activationfun = [np.vectorize(self._sigmoid) if i == 1 else np.vectorize(self._relu) if i == 2   else np.vectorize(self._tanh)   for i in activationfunc]
        self.activationfunderv = [np.vectorize(self._sigmoid_derv) if i == 1 else np.vectorize(self._relu_derv) if i == 2   else np.vectorize(self._tanh_derv)  for i in activationfunc]

#         #Test
#         self.weights = [np.array([[-0.2, -0.1],
#         [ 0.1,  0.3]]), np.array([[0.2],
#         [0.3]])]
#         self.bias = [np.array([0.1, 0.1]), np.array([0.2])]
        
        
        


    def weight_update(self):
        self.weights = [  w-w_u for w,w_u in zip(self.weights, weight_update) ]
    
    def bias_update(self): 
        self.bias = [  b-b_u for b,b_u in zip(self.bias, bias_update) ]

    
    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
        
    
    def _relu(self, x): 
        return max(0, x)
    
    def _tanh(self, x): 
         return (np.exp(x) - np.exp(-x))/ (np.exp(x)+ np.exp(-x))
        
        
    def _sigmoid_derv(self, x): 
             val = self._sigmoid(x) 
             return val * (1-val)
       

    def _relu_derv(self, x): 
        return 1 if x>0 else 0
    
    def _tanh_derv(self, x):
        return 1 - (((np.exp(x) - np.exp(-x))**2)/ ((np.exp(x)+ np.exp(-x))**2))

    def loss_func(self, x,y): 
        return (1/(len(x))) * sum(y-x)
    
    
    def cross_entropy_loss(self,pred, y):
        if y == 1:
            return -np.log(pred)
        else:
            return -np.log(1 - pred)
    
    def forward_propagate(self,inputs):

        activations = inputs
        net_inputs = []
        z_value = []
        activated_value = []
        activated_value.append(inputs)
        for w,b,a in zip(self.weights,self.bias,self.activationfun):
            net_inputs = (np.dot(activations, w)) + b
            
            activations = a(net_inputs)
            z_value.append(net_inputs)
            activated_value.append(activations)
       
        return z_value, activated_value
    

    def back_propogate(self,z_value, activated_value,  y):
#         eval = (1/(len(activated_value[-1]))) * sum(y-activated_value[-1])
        eval = self.lss(activated_value[-1],y)

        vfunc = np.vectorize(self._sigmoid_derv)
#         sig_derv_activ = [vfunc(x) for x in z_value]
        weight_update = []
        bias_update = []
        sig_derv_activ = []
        for i in range(len(self.weights)-1, -1,-1):
#                 sig_derv_activ[i] = sig_derv_activ[i] * eval
                sig_derv_activ.insert(0,(self.activationfunderv[i](z_value[i]))*eval)
                weight_update.insert(0,np.dot(np.reshape(sig_derv_activ[0], (-1, 1))
                                              ,np.reshape(activated_value[i],(-1,1)).transpose()).transpose()*self.alpha)
                bias_update.insert(0,sig_derv_activ[0] * self.alpha)
                eval = np.dot(np.reshape(sig_derv_activ[0], (-1, 1)).transpose(),self.weights[i].transpose())

        return weight_update, bias_update
    
        
        

if __name__ == "__main__":

    df = pd.read_csv('/Users/smiroshnikova/Desktop/trial.csv',header=None)
    df_Y = df.iloc[:,-1]
    df_X = df.iloc[:,0:-1]
    sample_len = len(df)
    """#inputs="[1,2,3] 0.25 [1,2,1] 1"
b=sys.argv[1]
c=b.split("[")
d=c[1].split("]")
e=d[0].split(",")
hidden=[]
e=['1','2','3']
for i in range (len(e)):
    hidden.append(int(e[i]))

    

learningrate=float(sys.argv[2])
b1=sys.argv[3]
f=b1.split("[")
g=f[1].split("]")
h=g[0].split(",")
activation=[]
for j in range (len(h)):
    activation.append(int(h[j]))

lossfunction=int(sys.argv[4])

print(hidden)
print(learningrate)
print(activation)
print(lossfunction)

"""
    mlp = MLP(numinputs=df_X.shape[1], hiddenlayers=[2], numoutputs=1,activationfunc = [1,1], alpha = 0.25, lossfunc =1 )
   
    for i in range(epochs):
        print('epoch:' +str(i))
        for i in range(sample_len):
            z_value, activated_value =mlp.forward_propagate(np.array(df_X.iloc[i]))
            weight_update, bias_update= mlp.back_propogate(z_value,activated_value, df_Y.iloc[i])
            mlp.weight_update()
            mlp.bias_update()
