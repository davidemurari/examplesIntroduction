import numpy as np
import matplotlib.pyplot as plt
import time as time_lib
import random

from tqdm import tqdm

import multiprocessing

from scripts.dynamics import vecField
from scripts.parareal import parallel_solver

def run_experiment(args,return_nets=False,verbose=False):
        
        system, nodes = args
        
        vecRef = vecField()
        number_processors = 1
        
        n_x = 5
        n_t = 2 #we do all the experiments with n_t = 2, which means we really just have coarse intervals
                #we do not further split them to simplify the problem. On the other hand, the code is flexible
                #also to this additional splitting.

        LB = -1.
        UB = 1.
        L = 5

        beta,gamma,N = vecRef.beta,vecRef.gamma,vecRef.N
        t_max = 100.
        num_t = 101
        L = 3
        vecRef.dt_fine = 1e-2
        
        time = np.linspace(0,t_max,num_t)
        
        dts = np.diff(time)
        
        y0 = np.array([0.3,0.5,0.2])

        weight = np.random.uniform(low=LB,high=UB,size=(L))
        bias = np.random.uniform(low=LB,high=UB,size=(L))

        data = {"vecRef":vecRef,
                "LB" : LB,
                "time" : time,
                "dts" : dts,
                "UB" : UB,
                "L" : L,
                "y0" : y0,
                "n_x" : n_x,
                "n_t" : n_t,
                "num_t" : num_t,
                "system" : system,
                "nodes" : nodes,
                "act_name" : "tanh",
                "number_processors":number_processors,
                "t_max":t_max,
                "weight":weight,
                "bias":bias}
                
        coarse_approx,networks,total_time,_,avg_coarse_step = parallel_solver(time,data,dts,vecRef,number_processors,verbose=verbose)

        if return_nets:
                return total_time,avg_coarse_step,coarse_approx,networks,data
        else:
                return total_time, avg_coarse_step