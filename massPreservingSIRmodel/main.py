import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import time
import scipy 
import os

from scripts.training import train_network
from scripts.utils import get_solutions
from scripts.networks import *
from scripts.generateData import generate, get_data_loaders
from scripts.plotting import plot

os.chdir("./MassPreservingSIRmodel")

torch.manual_seed(0)
np.random.seed(0)

dtype = torch.float64

#Parameters of the model
n = 3 #Input dimension
o = 3 #Output dimension

N = 5000
dim = n
Ntrain = 0.9
M = 6
T = 0.25
batch_size = 32

#Data generation
x_train, y_train, x_test, y_test, h = generate(N,M,T,Ntrain)
trainset,testset,trainloader,testloader = get_data_loaders(x_train,y_train,x_test,y_test,batch_size,dtype=dtype)

print(f"Time step is {h}")

#Definition of the two neural networks
net = Network(n, o, dtype)
netU = UnstructuredNetwork(n,o,5,dtype)

#Training of the neural network
criterion = nn.MSELoss()
lr = 5e-3
epochs = 300

optimizer = torch.optim.Adam(net.parameters(),lr=lr)
optimizerU = torch.optim.Adam(netU.parameters(),lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.45*epochs), gamma=0.1)
schedulerU = torch.optim.lr_scheduler.StepLR(optimizerU, step_size=int(0.45*epochs), gamma=0.1)

timestamp = time.strftime("%Y%m%d_%H%M%S") 

train = input("Do you want to train the constrained network? Type y if yes, another key otherwise\n")=="y"
if train:
    print("=====================================")
    print("Training of the constrained network")
    print("=====================================")
    train_network(net,trainloader,M,epochs,optimizer,criterion,scheduler)

    torch.save(net.state_dict(), f"trainedModels/constrainedNet_{timestamp}.pt")
else:
    net.load_state_dict(torch.load("trainedModels/constrainedNet.pt",map_location='cpu'))
    #net.load_state_dict(torch.load("trainedModels/constrainedNet_20240430_173429.pt",map_location='cpu'))


train = input("Do you want to train the unconstrained network? Type y if yes, another key otherwise\n")=="y"
if train:
    print("=====================================")
    print("Training of the unconstrained network")
    print("=====================================")
    train_network(netU,trainloader,M,epochs,optimizerU,criterion,schedulerU)
        
    torch.save(netU.state_dict(), f"trainedModels/unconstrainedNet_{timestamp}.pt")
else:
    netU.load_state_dict(torch.load("trainedModels/unconstrainedNet.pt",map_location='cpu'))
    #netU.load_state_dict(torch.load("trainedModels/unconstrainedNet_20240430_173429.pt",map_location='cpu'))

net.eval();
netU.eval();

data = next(iter(testloader))
input, output = data[0], data[1]
predSol = torch.zeros((len(input),dim,M-1),dtype=dtype)
predSolU = torch.zeros((len(input),dim,M-1),dtype=dtype)

predSol[:,:,0] = net(input)
predSolU[:,:,0] = netU(input)

for i in range(1,M-1):
    predSol[:,:,i] = net(predSol[:,:,i-1])
    predSolU[:,:,i] = netU(predSolU[:,:,i-1])
    
print("Test error: ",criterion(predSol,output))
print("Test error U : ",criterion(predSolU,output))
print("Sum first output: ",torch.sum(net(input[0:1])).item())
print("Sum first output U: ",torch.sum(netU(input[0:1])).item())
print("Sum first input: ",torch.sum(input[0:1]).item())

y0 = np.array([0.3,0.2,0.1])

it = 100
time_vec, sol_true, mat, matU = get_solutions(it,y0,net,netU,h,dtype)
plot(time_vec,sol_true,mat,matU)