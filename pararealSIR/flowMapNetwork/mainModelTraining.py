import numpy as np
import matplotlib.pyplot as plt
import torch
import time as time_lib
from scripts.network import network
from scripts.training import trainModel
from scripts.dynamics import vecField,vecField_np
import pickle 
import os

os.chdir("./pararealSIR/flowMapNetwork")

torch.manual_seed(1)
np.random.seed(1)
dtype=torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vecRef = vecField()

beta,gamma,N = vecRef.beta,vecRef.gamma,vecRef.N
t_max = 100.
num_t = 101
L = 3
n_t = 2
vecRef.dt_fine = 1e-2

y0 = np.array([0.3,0.5,0.2])

#Domain details
t0 = 0. #initial time
dt = 1. #time step for which we train the network
lb = 0.
ub = 1.
d = 3
    
n_train, epochs = 1000, int(1e5)
vec = vecField()
is_training = input("Do you want to train the network or load a pretrained model? Write y to train it\n")=="y"

lr = 5e-3

nlayers = 4
hidden_nodes = 10
act = "tanh"
dim_t = hidden_nodes
device = 'cpu'
bounds = [lb,ub]

model = network(neurons=hidden_nodes,d=d,dt=dt,act_name=act,nlayers=nlayers,dtype=dtype) #If you want a different network
model.to(device);

lr = 5e-3

tt = time_lib.time()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
timestamp = time_lib.strftime("%Y%m%d_%H%M%S") 
file_name = f"trained_model_SIR_{timestamp}"
path = f"trainedModels/{file_name}.pt"
Loss = trainModel(model,n_train,y0,dt,t_max,bounds,vec,epochs,device,dtype,optimizer)
torch.save(model.state_dict(), path)
model.eval();

training_time = time_lib.time() - tt

params = {
    "dt":dt,
    "n_train":n_train,
    "epochs":epochs,
    "nlayers":nlayers,
    "lr":lr,
    "act":act,
    "hidden_nodes":hidden_nodes,
    "dim_t":dim_t,
    "device":device,
    "file_name":file_name,
    "training_time":training_time
}

with open(f'trainedModels/{file_name}.pkl', 'wb') as f:
    pickle.dump(params, f)

print("Training time : ",training_time)