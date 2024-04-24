import torch
import numpy as np
from scipy.integrate import solve_ivp
import numpy as np

class vecField:
    def __init__(self):
        
        self.beta = 0.1
        self.gamma = 0.1
        self.N = 1.
        self.dt_fine = 0.
        self.d = 3
        
    def eval(self,y):
                
        if len(y.shape)==2:
            y1,y2,y3 = y[:,0:1],y[:,1:2],y[:,2:3]
            return torch.cat((
                -self.beta*y2*y1/self.N,
                self.beta*y1*y2/self.N - self.gamma*y2,
                self.gamma*y2
            ),dim=1)
        else:
            y1,y2,y3 = y[0],y[1],y[2]
            return torch.tensor([
                -self.beta*y2*y1/self.N,
                self.beta*y1*y2/self.N - self.gamma*y2,
                self.gamma*y2
            ])

    def residualLoss(self,z,z_d):
        return torch.mean((self.eval(z)-z_d)**2)
    
def solution_scipy(y0,t_eval,vec):
    
    #This method computes the approximate solution starting from y0
    #over the time interval t_eval of the vector field vec using a BDF
    #method from the Scipy library.

    print("Currently solving with Scipy integrator")
    
    def fun(t,y):
        y1,y2,y3 = y[0],y[1],y[2]
        return np.array([
                -vec.beta*y2*y1/vec.N,
                vec.beta*y1*y2/vec.N - vec.gamma*y2,
                vec.gamma*y2
            ])

    t_span = [t_eval[0],t_eval[-1]]
    res = solve_ivp(fun, t_span=t_span, t_eval=t_eval, y0=y0, method='RK45')
    return res.y

def approximate_solution(y0,model,time,dtype,vec,device):
    
    #This method evaluates the network solution on the entries of the time array
    #with initial condition stored in y0.
    
    print("Currently generating the network predictions")
    count = 1
    d = len(y0)
    dt = model.dt
    sol = torch.zeros((len(time),d),dtype=dtype).to(device)
    sol[0] = y0
    pos = 0
    for i in range(1,len(time)):
        t = time[i:i+1]
        if t>count*dt:
            count+=1
            pos = i-1
        if dtype==torch.float32:
            supp = model(sol[pos:pos+1],torch.from_numpy((t-dt*(count-1)).astype(np.float32)).to(device))
        else:
            supp = model(sol[pos:pos+1],torch.from_numpy((t-dt*(count-1))).to(device))
        sol[i:i+1] = supp
    return sol
    

import numpy as np

class vecField_np:
    def __init__(self,system="Rober"):
        self.system = system 
                
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