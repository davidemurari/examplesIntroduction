import torch
import numpy as np
import matplotlib.pyplot as plt
from scripts.utils import *
import matplotlib

#Setting the plotting parameters
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams['font.size']= 45
matplotlib.rcParams['font.family']= 'ptm' #'Times New Roman

def energyPlot(q0,pi0,tf,model,dtype,device):

    y0 = torch.cat((q0,pi0)).to(device)
        
    y0_np = torch.cat((q0,pi0)).detach().cpu().numpy()
    t_eval = np.linspace(0,tf,int(tf*10+1)) 

    sol_network = approximate_solution(y0,model,t_eval,dtype,device).detach().cpu().numpy().T
    sol_scipy = solution_scipy(y0_np,t_eval=t_eval)
        
    fig = plt.figure(figsize=(20,10))
    
    E0 = np.linalg.norm(y0,ord=2)**2/2
    EnergyVariationTrue = np.abs(np.linalg.norm(sol_scipy,ord=2,axis=0)**2/2 - E0)
    EnergyVariationPred = np.abs(np.linalg.norm(sol_network,ord=2,axis=0)**2/2 - E0)
    
    plt.semilogy(t_eval,EnergyVariationTrue,'-',c=f'k',label=fr'$|E(t)-E(0)|$ reference',linewidth=5,)
    plt.semilogy(t_eval,EnergyVariationPred,'-',c=f'b',label=fr'$|E(t)-E(0)|$ pred',linewidth=5)
     
    plt.legend(fontsize=25,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel(r"$t$")
    plt.ylabel(r"$|E(t)-E(0)|$")
    plt.savefig(f"savedPlots/energy_variation_T{tf}.pdf",bbox_inches='tight')
    plt.show();      

def testPlot(q0,pi0,tf,model,dtype,device):

    y0 = torch.cat((q0,pi0)).to(device)
        
    y0_np = torch.cat((q0,pi0)).detach().cpu().numpy()
    t_eval = np.linspace(0,tf,int(tf*10+1)) 

    sol_network = approximate_solution(y0,model,t_eval,dtype,device).detach().cpu().numpy().T
    sol_scipy = solution_scipy(y0_np,t_eval=t_eval)
    
    fig = plt.figure(figsize=(20,10))
    
    plt.plot(t_eval,sol_scipy[0],'-',c=f'k',label=fr'$q$ reference',linewidth=5,)
    plt.plot(t_eval,sol_scipy[1],'-',c=f'b',label=fr'$p$ reference',linewidth=5)

    plt.plot(t_eval,sol_network[0],'--',c='y',label=fr'$q$ pred',linewidth=5,)
    plt.plot(t_eval,sol_network[1],'--',c='magenta',label=fr'$p$ pred',linewidth=5,)
     
    plt.legend(fontsize=25,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel(r"$t$")
    plt.ylabel("Solution")
    plt.show();