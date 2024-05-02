import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class DynamicBlock(nn.Module):
    def __init__(self, nlayer, input, lift, dtype):
        super(DynamicBlock, self).__init__()
        
        self.nlayers = nlayer
        self.input = input
        self.lift = lift
        self.dtype = dtype
        
        self.DOF = int(self.input * (self.input-1) / 2)

        self.A = nn.ModuleList([nn.Linear(self.input,self.lift,bias=False,dtype = self.dtype) for i in range(self.nlayers)])
        self.B = nn.ModuleList([nn.Linear(self.lift,self.DOF,dtype = self.dtype) for i in range(self.nlayers)])
        self.Biases = nn.ParameterList([nn.Parameter(torch.randn(self.lift,dtype = self.dtype)) for i in range(self.nlayers)])
        self.nl = nn.LeakyReLU()

        self.dts = nn.Parameter(torch.rand(self.nlayers,dtype = self.dtype))

        self.alpha = torch.ones(self.input,dtype = self.dtype)

    def buildSkew(self,ff):
        res = torch.zeros((len(ff),self.input,self.input),dtype = self.dtype)
        iu1 = torch.triu_indices(self.input,self.input,1)
        res[:,iu1[0],iu1[1]] = ff
        res = res - torch.transpose(res,1,2)
        return res

    def forward(self, x):
        
        for i in range(self.nlayers):
            c = self.A[i](x)+self.Biases[i]
            ff = self.nl(self.B[i](self.nl(c)))
            Mat = self.buildSkew(ff)
            x = x + 0.1 * self.alpha @ Mat
        return x

#Linear lifting layer that preserves the sum
class Lift(nn.Module):
    def __init__(self, lift,dtype):
        super(Lift, self).__init__()
        self.outputDim = lift
        self.dtype = dtype
    def forward(self,x):
        I = torch.eye(x.shape[1],dtype = self.dtype)
        Z = torch.zeros((x.shape[1],self.outputDim-x.shape[1]),dtype = self.dtype)
        liftMat = torch.cat((I,Z),dim=1)
        return x@liftMat

#Projection layer that preserves the sum
class Projection(nn.Module):
    def __init__(self, inputDim, lowerDim, dtype):
        super(Projection, self).__init__()
        self.outputDim = lowerDim
        self.input = inputDim
        self.dtype = dtype
        
    def forward(self,x):
        I = torch.eye(self.outputDim,dtype = self.dtype)
        Z = torch.zeros((self.input-self.outputDim,self.outputDim),dtype = self.dtype)
        projMat = torch.cat((I,Z),dim=0)
        s = torch.sum(x[:,self.outputDim:],dim=1).unsqueeze(1)/(self.outputDim)
        y = x @ projMat
        y = y + s
        return y        

class Network(nn.Module):
    def __init__(self, input, output, dtype):
        super(Network, self).__init__()
        
        self.input = input
        self.output = output
        self.dtype = dtype
        
        dim = [3,10,10,15]
        lista = []
        for i in np.arange(1,len(dim)):
            if dim[i]>dim[i-1]:
                lista.append(Lift(dim[i],dtype = self.dtype))
            else:
                lista.append(Projection(dim[i-1],dim[i],dtype = self.dtype))
                
            #lista.append(DynamicBlock(3,dim[i],max(dim)))
            lista.append(DynamicBlock(3,dim[i],50,dtype = self.dtype))

        lista.append(Projection(dim[-1],self.output,dtype = self.dtype))
        self.seq = nn.Sequential(*lista)

    def forward(self, x):
        x = self.seq(x)
        return x
    
class UnstructuredNetwork(nn.Module):
    def __init__(self, input, output, nlayers, dtype):
        super(UnstructuredNetwork, self).__init__()
        
        self.input = input
        self.output = output
        
        self.dtype = dtype
        
        self.nlayers = nlayers
        
        self.lift = nn.Linear(input,20,dtype=dtype)
        self.linears = nn.ModuleList([nn.Linear(20,20,dtype=dtype) for _ in range(self.nlayers)])
        self.linearsB = nn.ModuleList([nn.Linear(20,20,dtype=dtype) for _ in range(self.nlayers)])
        self.proj = nn.Linear(20,output,dtype=dtype)
        
        self.dts = nn.Parameter(torch.randn(self.nlayers,dtype=dtype))

        self.nl = nn.LeakyReLU()

    def forward(self, x):
        x = self.lift(x)
        for i in range(self.nlayers):
            x = x + self.dts[i] * self.nl(self.linearsB[i](self.nl(self.linears[i](x))))
        return self.proj(x)
