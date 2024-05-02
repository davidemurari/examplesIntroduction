import numpy as np
import matplotlib.pyplot as plt
import time as time_lib
import matplotlib

#Setting the plotting parameters
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams['font.size']= 45
matplotlib.rcParams['font.family']= 'ptm' #'Times New Roman

def plot_results(y0,coarse_approx,networks,system,time_plot,time_plot_sequential,output,network_sol,list_of_labels,total_time,time,n_x,n_t,L,number_iterates=1,btype=None,node_type=None):

    fig = plt.figure(figsize=(20,10),dpi=600)
    
    for i in range(len(y0)):
        if i==1:
            plt.plot(time_plot_sequential,output[:,i],'-',label=f"{list_of_labels[i]} reference",linewidth=5)
            plt.plot(time_plot,network_sol[i],'--',label=f"{list_of_labels[i]} parareal",linewidth=5)
        else:
            plt.plot(time_plot_sequential,output[:,i],'-',label=f"{list_of_labels[i]} reference",linewidth=5)
            plt.plot(time_plot,network_sol[i],'--',label=f"{list_of_labels[i]} parareal",linewidth=5)

    plt.legend(fontsize=25)
    
    title = f"Comparison of solutions, $C$={n_x}, $H$={L},\n Median computational time over {number_iterates} iterates: {np.round(total_time,2)}s" if number_iterates>1 \
            else f"Comparison of solutions, $C$={n_x}, $H$={L},\n Computational time: {np.round(total_time,2)}s"
    plt.title(title) 
    plt.xlabel(r'$t$')
    name_plot = f"savedPlots/ELM_pararealPlot_{system}.pdf"
    
    if node_type!=None:
        name_plot = f"savedPlots/ELM_pararealPlot_{system}_{node_type}.pdf"
        
    plt.savefig(name_plot,bbox_inches='tight')
    #plt.show();