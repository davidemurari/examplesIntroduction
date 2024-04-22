import numpy as np
import torch
import torch.nn as nn
from scripts.utils import vecField

#Custom activation functions   
class sinAct(nn.Module):
        def __init__(self):
            super().__init__()  
        def forward(self,x):
            return torch.sin(x) 

class swishAct(nn.Module):
        def __init__(self):
            super().__init__()  
        def forward(self,x):
            return torch.sigmoid(x)*x
        
class integralTanh(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,x):
            return torch.log(torch.cosh(x)) #this is the primitive of tanh(x)


class network(nn.Module):
        def __init__(self,neurons=100,dt=1.,act_name='tanh',d=4,nlayers=3,dtype=torch.float):
            super().__init__()
            
            ##The outputs of this network are to be intended as the variables
            ##q1,q2,pi1,pi2, i.e. the physical configuration variables together
            ##with the non-conservative momenta.
            
            torch.manual_seed(1)
            np.random.seed(1)
            
            self.dtype = dtype
            self.vec = vecField()
            self.dt = dt
            
            activations = {
                "tanh": nn.Tanh(),
                "sin": sinAct(),
                "sigmoid": nn.Sigmoid(),
                "swish": swishAct(),
                "intTanh":integralTanh()
            }
            
            self.act = activations[act_name]
            
            self.H = neurons
            self.d = d
            self.nlayers = nlayers
            
            self.lift = nn.Linear(self.d+1,self.H,dtype=dtype)
            self.linears = nn.ModuleList([nn.Linear(self.H,self.H,dtype=dtype) for _ in range(self.nlayers)])
              
            self.proj = nn.Linear(self.H,self.d,dtype=dtype)
            
            self.f = lambda t: 1-torch.exp(-t)
        
        def parametric(self,y0,t):
            
            input = torch.cat((y0,t),dim=1)
            
            H = self.act(self.lift(input))
            for i in range(self.nlayers):
                H = H + self.act(self.linears[i](H))
            res = self.proj(H)
            
            return res     
        def forward(self,y0,t):
            
            y0 = y0.reshape(-1,self.d)
            t = t.reshape(-1,1)
            
            t0 = torch.zeros_like(t)
            res = self.parametric(y0,t)
            res0 = self.parametric(y0,t0)
            update = res - res0
            res = y0 + update
            return res 
        
        