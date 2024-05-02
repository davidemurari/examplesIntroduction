import time as time_lib
import torch
import os

from scripts.networks import network
from scripts.utils import *
from scripts.training import trainModel
from scripts.plotting import *

os.chdir("./harmonicOscillator")

dtype=torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Domain details
t0 = 0. #initial time
tf = 100. #final time
dt = 1 #time step for which we train the network

qlb = -1.2 #lower bound values for q
qub = 1.2 #upper bound values for q
plb = -1.2 
pub = 1.2
q0 = torch.tensor([1.],dtype=dtype)
pi0 = torch.tensor([0.],dtype=dtype)
d = 2

n_train, epochs = 1000, int(5e4)
vec = vecField()

nlayers = 2
hidden_nodes = 10
act = "tanh"
dim_t = hidden_nodes
bounds = [t0,tf,qlb,qub,plb,pub]

is_training = input("Do you want to train the network or load a pretrained model? Write y to train it\n")=="y"

model = network(neurons=hidden_nodes,d=d,dt=dt,act_name=act,nlayers=nlayers,dtype=dtype) #If you want a different network
model.to(device);

lr = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if is_training:
    Loss = trainModel(model,n_train,q0,pi0,dt,bounds,vec,epochs,device,dtype,optimizer,verbose=0)
    #Path to save the model
    timestamp = time_lib.strftime("%Y%m%d_%H%M%S") 
    path = f"trainedModels/trained_model_{timestamp}.pt"
    torch.save(model.state_dict(), path)
    model.load_state_dict(torch.load("trainedModels/best.pt",map_location=device))
else:
    path = input("Write the name of the model to load, do not include .pt\n")
    model.load_state_dict(torch.load("trainedModels/"+path+".pt",map_location=device))
model.eval();
model.to('cpu');

t_max_list = [100,500]

for t_max in t_max_list:
    energyPlot(q0,pi0,t_max,model,dtype,device)
    testPlot(q0,pi0,t_max,model,dtype,device)