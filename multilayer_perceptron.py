import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
#import sys
class MLP():

    def __init__(self, numinputs=4, hiddenlayers=[4,2,2,1], numoutputs=1, activationfunc = [1,1,1], alpha = 0.25, lossfunc = 1,cost_func, x_low, x_high, size=50):
        self.numinputs = numinputs
        self.hiddenlayers = hiddenlayers
        self.numoutputs = numoutputs
#new variables for ParticleSwarm
        self.cost_func = cost_func
        self.x_low = np.array(x_low)
        self.x_high = np.array(x_high)
        self.v_high = (self.x_high - self.x_low) / 2.
        self.v_low = -self.v_high
        self.dim = len(x_low)
        self.size = size
        self.X = np.random.uniform(self.x_low, self.x_high, (self.size, self.dim))
        self.V = np.random.uniform(self.v_low, self.v_high, (self.size, self.dim))
        self.P = self.X.copy()
        self.S = self.cost_func(self.X)
        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()
# end of ParticleSwarm variables
        # create a generic representation of the layers
        layers = [numinputs] + hiddenlayers + [numoutputs]
         
        # create random connection weights for the layers
        weights = []
        bias = []
        for i in range(len(layers)-1):
            bias.append(np.ones(layers[i+1]))
            weights.append(np.random.rand(layers[i], layers[i+1]))
        self.weights = weights
        self.bias = bias
        self.alpha = alpha
        
        
        if lossfunc == 1: 
            self.lss = self.loss_func
        elif lossfunc == 2: 
            self.lss = self.cross_entropy_loss
    
        self.activationfun = [np.vectorize(self._sigmoid) if i == 1 else np.vectorize(self._relu) if i == 2   else np.vectorize(self._tanh)   for i in activationfunc]
        self.activationfunderv = [np.vectorize(self._sigmoid_derv) if i == 1 else np.vectorize(self._relu_derv) if i == 2   else np.vectorize(self._tanh_derv)  for i in activationfunc]

# #         #Test
#         self.weights = [np.array([[-0.2, -0.1],
#         [ 0.2,  0.3]]), np.array([[0.2],
#         [0.3]])]
#         self.bias = [np.array([0.1, 0.1]), np.array([0.2])]
        
        
        


    def weight_update(self,weight_update):
        self.weights = [  w-w_u for w,w_u in zip(self.weights, weight_update) ]
        
    
    def bias_update(self,bias_update): 
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
        #add aquare root
        return y-x
    
    
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
    

    def back_propogate(self,z_value, activated_value,  y,step):
        eval =  self.lss(activated_value[-1],y)
        weight_update = []
        bias_update = []
        sig_derv_activ = []
        for i in range(len(self.weights)-1, -1,-1):
                
                sig_derv_activ = self.activationfunderv[i](z_value[i])*eval
                weight_updt = self.alpha* (1.0/step)*np.sum(np.array([np.dot(np.reshape(sig_derv_activ[x],(-1,1)),np.reshape(activated_value[i][x],(-1,1)).transpose()) for x in range((len(activated_value[i])))]),axis=0)

                bias_updt = np.average(sig_derv_activ,axis = 0)* self.alpha
                eval = [np.dot(np.reshape(sig_derv_activ[x], (-1, 1)).transpose(),self.weights[i].transpose()) [0]for x in range(len(sig_derv_activ))]
                weight_update.insert(0,weight_updt.transpose())
                bias_update.insert(0,bias_updt)
        return weight_update, bias_update
        
        
        
    def predict_values(self, inputs, y): 
        z_value, activated_value = mlp.forward_propagate(np.array(inputs))
        loss = self.lss(activated_value[-1],np.reshape(np.array(y),(-1,1)))
        cost = (1.0/len(y))*sum(loss)
        print('#########')
        print('Cost:'+str(cost))
        return activated_value[-1],loss, cost

def training(df_X,hiddenlayers,numoutputs,activationfunc,alpha,loss,step, epochs):
    print('Activation Functions are as follows: 1 -> Sigmoid | 2-> ReLu | 3-> tanH')
    print('Loss Functions are as follows: 1 -> Error | 2-> Cross Entropy')
    print('#########')
    print('Results for hyperparameter:')
    print('Epochs:' +str(epochs)) 
    print('Steps:' + str(step))
    print('Learning rate:' +str(alpha))
    print('Number of hidden layers:'+str(len(hiddenlayers)) )
    print('Loss function:'+str(lossfunc))
    print('Activation functions:'+str(activationfunc))
    mlp = MLP(df_X.shape[1], hiddenlayers, numoutputs,activationfunc, alpha, lossfunc)

     # Need to add this as arg as well and epochs
    for i in range(epochs):
        for i in range(0, sample_len,step): #sample_len,5): 
            z_value, activated_value =mlp.forward_propagate(np.array(X_train.iloc[i:i+step]))            
            weight_update, bias_update= mlp.back_propogate(z_value,activated_value, np.array(Y_train.iloc[i:i+step]).reshape(step,1),step)
            mlp.weight_update(weight_update)
            mlp.bias_update(bias_update)
    return mlp


 #ParticleSwarm code           
    def optimize(self, epsilon=1e-3, max_iter=100):
        iteration = 0
        while self.best_score > epsilon and iteration < max_iter:
            self.update()
            iteration = iteration + 1
        return self.g

    def update(self, omega=1.0, phi_p=2.0, phi_g=2.0):
        # Velocities update
        R_p = np.random.uniform(size=(self.size, self.dim))
        R_g = np.random.uniform(size=(self.size, self.dim))

        self.V = omega * self.V \
                + phi_p * R_p * (self.P - self.X) \
                + phi_g * R_g * (self.g - self.X)

        # Velocities bounding
        self.V = np.where(self.V < self.v_low, self.v_low, self.V)
        self.V = np.where(self.V > self.v_high, self.v_high, self.V)

        # Positions update
        self.X = self.X + self.V

        # Positions bounding
        self.X = np.where(self.X < self.x_low, self.x_low, self.X)
        self.X = np.where(self.X > self.x_high, self.x_high, self.X)

        # Best scores
        scores = self.cost_func(self.X)

        better_scores_idx = scores < self.S
        self.P[better_scores_idx] = self.X[better_scores_idx]
        self.S[better_scores_idx] = scores[better_scores_idx]

        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()


    def evaluate_simple(x):
 #   """Simple 2D concave surface with the minimum at the origin"""
       x = np.reshape(x, (-1, 2))
       return np.sum(x**2, axis=1)

    def schaffer6(x):
#    """Complex 2D surface with the global minimum at the origin"""
      x = np.reshape(x, (-1, 2))
      sumOfSquares = np.sum(x**2, axis=1)
      return 0.5 + (np.sin(np.sqrt(sumOfSquares))**2 - 0.5) / (1 + 0.001 * sumOfSquares)**2

    def test_simple_optimization():
        swarm = pso.ParticleSwarm(
            cost_func=evaluate_simple,
            x_low=[-100., -100.],
            x_high=[100., 100.],
            size=20
    )

    best = swarm.optimize(epsilon=1e-6, max_iter=100000)
    assert evaluate_simple(best) < 1e-6

    def test_complex_optimization():
        swarm = pso.ParticleSwarm(
            cost_func=schaffer6,
            x_low=[-100., -100.],
            x_high=[100., 100.],
            size=20
        )

        best = swarm.optimize(epsilon=1e-6, max_iter=100000)
        assert schaffer6(best) < 1e-6

if __name__ == "__main__":

#     df = pd.read_csv('/Users/smiroshnikova/Desktop/trial.csv',header=None)
    df = pd.read_csv('/Users/smiroshnikova/Desktop/data_banknote_authentication.csv',header=None)

    df_Y = df.iloc[:,-1]
    df_X = df.iloc[:,0:-1]
    X_train,X_test,Y_train,Y_test=train_test_split(df_X,df_Y,test_size=0.5)
    sample_len = len(X_train)
    epochs = 1
    #"[1,2,3] 0.25 [1,2,1] 1" input format
    """b=sys.argv[1]
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
    hiddenlayers=[2]
    numoutputs=1
    activationfunc = [1,1]
    alpha = 0.25
    lossfunc =1
    step = 2 # Need to add this as arg as well and epochs
    epochs = 1000
    mlp = training(X_train, hiddenlayers, numoutputs,activationfunc , alpha, lossfunc, step, epochs)
    pred_val,loss, cost = mlp.predict_values(X_test,Y_test)
    
    epoch_test = []
    for i in range(0, 10000, 5000): 
        mlp = training(X_train, hiddenlayers, numoutputs,activationfunc , alpha, lossfunc, step, i)
        pred_val,loss, cost = mlp.predict_values(X_test,Y_test)
        epoch_test.append((i,cost))
    
    
    
