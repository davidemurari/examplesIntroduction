from scipy.integrate import solve_ivp
from scripts.dynamics import *
import time as time_lib
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator as linOp
from scipy.sparse import diags, eye, kron

import random
import numpy as np


def act(t,w,b,act_name="Tanh"):
    if act_name=="Tanh":
        return np.tanh(w*t+b), (1-np.tanh(w*t+b)**2)*w #function at the node, derivative at the node    
    elif act_name=="Sigmoid":
        func = lambda x : 1/(1+np.exp(-x))
        func_p = lambda x : func(x)*(1-func(x))
        return func(w*t+b), func_p(w*t+b)*w
    else: #sin
        func = lambda x : np.sin(x)
        func_p = lambda x: np.cos(x)
        return func(w*t+b), func_p(w*t+b)*w

#https://mathworld.wolfram.com/LobattoQuadrature.html
def lobattoPoints(n):
    #Nodes in [-1,1]
    if n==3:
        nodes = np.array([-1,0.,1.])
    elif n==4:
        nodes = np.array([-1,-np.sqrt(5)/5,np.sqrt(5)/5,1.])
    elif n==5:
        nodes = np.array([-1,-np.sqrt(21)/7,0.,np.sqrt(21)/7,1.])
    #Transforming them into [0,1]
    return nodes/2+0.5

#https://mathworld.wolfram.com/LobattoQuadrature.html
def legendrePoints(n):
    #Nodes in [-1,1]
    if n==2:
        nodes = np.array([-np.sqrt(3)/3,np.sqrt(3)/3])
    elif n==3:
        nodes = np.array([-np.sqrt(15)/5,0.,np.sqrt(15)/5])
    elif n==4:
        p1 = 1/35*np.sqrt(525+70*np.sqrt(3))
        p2 = 1/35*np.sqrt(525-70*np.sqrt(3))
        nodes = np.array([-p1,-p2,p2,p1])
    elif n==5:
        p1 = 1/21 * np.sqrt(245+14*np.sqrt(70))
        p2 = 1/21 * np.sqrt(245-14*np.sqrt(70))
        nodes = np.array([-p1,-p2,0.,p2,p1])
    #Transforming them into [0,1]
    return nodes/2+0.5

def uniformPoints(n):
    return np.linspace(0,1,n)
    
class flowMap:
    def __init__(self,y0,initial_proj,weight,bias,dt=1,n_t=5,n_x=10,L=10,LB=-1.,UB=1.,act_name="tanh",nodes="uniform",verbose=False):
        
        self.vec = vecField()
        self.act = lambda t,w,b : act(t,w,b,act_name=act_name)
        self.dt = dt
        self.d = len(y0) #dimension phase space
        
        self.verbose = verbose
        
        self.n_x = n_x
        self.h = np.zeros((n_x,L))
        self.hd = np.zeros((n_x,L))
        
        self.iter = 0
        
        self.y0 = y0
        self.y0_supp = y0
        
        self.L = L #number of neurons
        self.LB = LB #Lower boundary for weight and bias sampling 
        self.UB = UB #Upper boundary for weight and bias sampling
        
        if len(weight)==0:
            self.weight = np.random.uniform(low=self.LB,high=self.UB,size=(self.L))
        else:
            self.weight = weight
        if len(bias)==0:
            self.bias = np.random.uniform(low=self.LB,high=self.UB,size=(self.L))
        else:
            self.bias = bias
        
        self.n_t = n_t
        self.t_tot = np.linspace(0,dt,self.n_t)
        if nodes=="uniform":
            self.x = uniformPoints(self.n_x)
        elif nodes=="legendre":
            self.x = legendrePoints(self.n_x)
        elif nodes=="lobatto":
            self.x = lobattoPoints(self.n_x)

        for i in range(n_x):
            self.h[i], self.hd[i] = self.act(self.x[i],self.weight,self.bias)

        self.h0 = self.h[0] #at the initial time, i.e. at x=0.
        self.hd0 = self.hd[0]
        
        self.computational_time = None        
        self.computed_projection_matrices = np.tile(initial_proj,(self.n_t-1,1)) #one per time subintreval       
        self.computed_initial_conditions = np.zeros((self.n_t-1,self.d)) #one per time subinterval
        self.training_err_vec = np.zeros((self.n_t,1))
        self.sol = np.zeros((self.n_t,self.d))
    
    def to_mat(self,y,a,b):
        return y.reshape((a,b),order='F')
    
    def residual(self,c_i,xi_i):
        
        y = (self.h-self.h0)@self.to_mat(xi_i,self.L,self.d) + self.y0_supp.reshape(1,-1)
        y_dot = c_i * self.hd @ self.to_mat(xi_i,self.L,self.d)
        
        vecValue = self.vec.eval(y)
        Loss = (y_dot - vecValue)
        return Loss.reshape(-1,order='F')
    
    def re(self,a,H):
        return np.einsum('i,ij->ij',a,H)

    def jac_residual(self,c_i,xi_i):
        
        np.random.seed(17)
        random.seed(17)

        H = self.h - self.h0    
        weight = xi_i

        W = self.to_mat(weight,self.L,self.d)
        y = H@W + self.y0_supp.reshape(1,-1)
                
        y1,y2,y3 = y[:,0],y[:,1],y[:,2]
        zz = np.zeros((self.n_x,self.L))
        beta,gamma,N = self.vec.beta, self.vec.gamma, self.vec.N
        row1 = np.concatenate((c_i*self.hd+beta*self.re(y2,H)/N,beta*self.re(y1,H)/N,zz),axis=1)
        row2 = np.concatenate((-beta*self.re(y2,H)/N,c_i*self.hd-beta*self.re(y1,H)/N+gamma*H,zz),axis=1)
        row3 = np.concatenate((zz,-gamma*H,c_i*self.hd),axis=1)
        return np.concatenate((row1,row2,row3),axis=0)
    
    def approximate_flow_map(self):
        
        np.random.seed(17)
        random.seed(17)
        
        self.training_err_vec[0] = 0.        
        self.sol[0] = self.y0_supp
        
        initial_time = time_lib.time()
        
        for i in range(self.n_t-1):
            
            self.iter = 1
            
            c_i = (self.x[-1]-self.x[0]) / (self.t_tot[i+1]-self.t_tot[i])
            xi_i = self.computed_projection_matrices[i] 
            self.computed_initial_conditions[i] = self.y0_supp
            
            func = lambda x : self.residual(c_i,x)
            jac = lambda x : self.jac_residual(c_i,x)
            self.computed_projection_matrices[i] = least_squares(func,x0=xi_i,verbose=0,xtol=1e-5,method='lm',jac=jac).x
            Loss = func(self.computed_projection_matrices[i])
                
            y = (self.h-self.h0)@self.to_mat(self.computed_projection_matrices[i],self.L,self.d) + self.y0_supp.reshape(1,-1)
            self.y0_supp = y[-1]
            self.sol[i+1] = self.y0_supp
            self.training_err_vec[i+1] = np.sqrt(np.mean(Loss**2))
        final_time = time_lib.time()
        
        self.computational_time = final_time-initial_time
        if self.verbose:
            print(f"Training complete. Required time {self.computational_time}")
    
    def analyticalApproximateSolution(self,t):
        
        j = np.searchsorted(self.t_tot,t,side='left') #determines the index of the largest number in t_tot that is smaller than t
        #In other words, it finds where to place t in t_tot in order to preserve its increasing ordering
        j = j if j>0 else 1 #so if t=0 we still place it after the first 0.
        
        y_0 = self.sol[j-1]        
        x = np.array([(t - self.t_tot[j-1]) / (self.t_tot[j]-self.t_tot[j-1])])
        h,_ = act(x,self.weight,self.bias)
        h0,_ = act(0*x,self.weight,self.bias)
        y = self.to_mat(self.computed_projection_matrices[j-1],self.L,self.d).T@(h-h0) + y_0
        return y
    
    def plotOverTimeRange(self,time):
        sol_approximation = np.zeros((self.d,len(time)))
        for i,t in enumerate(time):
            sol_approximation[:,i] = self.analyticalApproximateSolution(t)
        return sol_approximation