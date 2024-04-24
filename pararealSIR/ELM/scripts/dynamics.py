import numpy as np

class vecField:
    def __init__(self):
           
        self.beta = 0.1
        self.gamma = 0.1
        self.N = 1.
        self.dt_fine = 0.
       
    def eval(self,y):
        
        if len(y.shape)==2:
            y1,y2,y3 = y[:,0:1],y[:,1:2],y[:,2:3]
            return np.concatenate((
                -self.beta*y2*y1/self.N,
                self.beta*y1*y2/self.N - self.gamma*y2,
                self.gamma*y2
            ),axis=1)
        else:
            y1,y2,y3 = y[0],y[1],y[2]
            return np.array([
                -self.beta*y2*y1/self.N,
                self.beta*y1*y2/self.N - self.gamma*y2,
                self.gamma*y2
            ])
        
        