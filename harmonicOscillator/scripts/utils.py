import torch
import numpy as np
from scipy.integrate import solve_ivp

class vecField:
    def __init__(self):
                
        self.d = 2
        
    def eval(self,x,p):
        #This method implements the right-hand-side of the ODE we work with.
        return torch.cat((p,
                    -x),dim=1)

    def residualLoss(self,z,z_d,t):
        d = self.d
        x = z[:,:d//2]
        pi  = z[:,d//2:]
        return torch.mean((self.eval(x,pi)-z_d)**2)
 
def solution_scipy(y0,t_eval):    
    def fun(t,y):
        q,p = y[0],y[1]
        return np.array([p,-q])

    t_span = [t_eval[0],t_eval[-1]]
    res = solve_ivp(fun, t_span=t_span, t_eval=t_eval, y0=y0, method='RK45')
    return res.y

def approximate_solution(y0,model,time,dtype,device):
    
    #This method evaluates the network solution on the entries of the time array
    #with initial condition stored in y0.
    
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
    

