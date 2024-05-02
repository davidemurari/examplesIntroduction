import numpy as np
import matplotlib.pyplot as plt
import torch
import time as time_lib
import os
from scipy.integrate import solve_ivp
import pickle

from scripts.network import network
from scripts.training import trainModel
from scripts.dynamics import vecField,vecField_np
from scripts.parareal import parallel_solver
from scripts.ode_solvers import solver
from scripts.plotting import plot_results

os.chdir("./pararealSIR/flowMapNetwork")

torch.manual_seed(1)
np.random.seed(1)
dtype=torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vecRef = vecField_np()

beta,gamma,N = vecRef.beta,vecRef.gamma,vecRef.N
t_max = 100.
num_t = 101
L = 3
n_t = 2
vecRef.dt_fine = 1e-2
y0 = np.array([0.3,0.5,0.2])
d = 3

#Domain details
t0 = 0. #initial time

with open('trainedModels/trained_model_SIR_20240424_161900.pkl', 'rb') as f:
    params = pickle.load(f)
    
dt = params["dt"]
nlayers = params["nlayers"]
hidden_nodes = params["hidden_nodes"]
act = params["act"]
dim_t = params["dim_t"]
device = params["device"]
file_name = params["file_name"]

model = network(neurons=hidden_nodes,d=d,dt=dt,act_name=act,nlayers=nlayers,dtype=dtype) #If you want a different network
model.to(device);


working_directory = os.getcwd()
print(working_directory)
model.load_state_dict(torch.load(f"trainedModels/{file_name}.pt",map_location=device))

model.eval();

data = {"y0" : y0,
        "device":device,
        "dtype":dtype}

number_points = int(t_max / dt)
time = np.linspace(0,100,number_points+1)
dts = np.diff(time)

num_processors_list = [1]#np.arange(1,8)
computational_times = []
overheads = []
used_processors = []
for number_processors in num_processors_list:
    print(f"Experiment with {number_processors} processors")
    coarse_values_parareal,networks, total_time, number_processors, overhead_costs = parallel_solver(time,data,dts,vecRef,number_processors,model,verbose=True)
    used_processors.append(number_processors)
    overheads.append(overhead_costs)
    computational_times.append(total_time)

def get_detailed_solution():
    num_steps = int((dts[0] / vecRef.dt_fine))
    time_plot = np.linspace(0,dts[0],num_steps)
    sol = model.plotOverTimeRange(y0,time_plot,data)
    total_time = time_plot
    for i in np.arange(len(dts)):
        num_steps = int((dts[i] / vecRef.dt_fine))
        time_plot = np.linspace(0,dts[i],num_steps)[1:]
        sol = np.concatenate((sol,model.plotOverTimeRange(coarse_values_parareal[i+1],time_plot,data)),axis=1)
        total_time = np.concatenate((total_time,time_plot+total_time[-1]),axis=0)
    return sol,total_time

network_sol, time_plot = get_detailed_solution()

initial = time_lib.time()
arg = [[y0,time_plot[-1],time_plot],vecRef]
output,time_plot_sequential = solver(arg,final=False)
#output, _ = dop853(funcptr, y0, t_eval=time_plot, rtol=1e-11, atol=1e-10)
final = time_lib.time()
print(f"Computational time sequential: {final-initial}")
print(f"Computational time parallel with {number_processors} processors: {total_time}")

if len(y0)==2:
        list_of_labels = [r"$\mathbf{x}_1$",r"$\mathbf{x}_2$"]
elif len(y0)==3:
        list_of_labels = [r"$\mathbf{x}_1$",r"$\mathbf{x}_2$",r"$\mathbf{x}_3$"]
elif len(y0)==4:
        list_of_labels = [r"$\mathbf{x}_1$",r"$\mathbf{x}_1'$",r"$\mathbf{x}_2$",r"$\mathbf{x}_2'$"]
else:
        list_of_labels = []

plot_results(y0,time_plot,time_plot_sequential,output,network_sol,list_of_labels,total_time,time)