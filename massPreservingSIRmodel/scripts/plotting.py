import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot(time,sol_true,mat,matU):
    # Create subplots with shared x-axis and set the figure size
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(25, 25),dpi=600)


    ax1.plot(time,sol_true[0,:],'-',c="blue",label=r"$x$ ref.",linewidth=6)
    ax1.plot(time,sol_true[1,:],'-',c="green",label=r"$y$ ref.",linewidth=6)
    ax1.plot(time,sol_true[2,:],'-',c="purple",label=r"$z$ ref.",linewidth=6)

    ax1.plot(time,mat[0,:],'-o',c="cyan",label=r"$x$ constr.",linewidth=3,markersize=8)#,markevery=3)#,alpha=0.7)
    ax1.plot(time,mat[1,:],'-o',c="gold",label=r"$y$ constr.",linewidth=3,markersize=8)#,markevery=3)#,alpha=0.7)
    ax1.plot(time,mat[2,:],'-o',c="chocolate",label=r"$z$ constr.",linewidth=3,markersize=8)#,markevery=3)#,alpha=0.7)

    ax1.plot(time,matU[0,:],'-o',c="magenta",label=r"$x$ unconstr.",linewidth=1,markersize=4)#,markevery=3)
    ax1.plot(time,matU[1,:],'-o',c="black",label=r"$y$ unconstr.",linewidth=1,markersize=4)#,markevery=3)
    ax1.plot(time,matU[2,:],'-o',c="lime",label=r"$z$ unconstr.",linewidth=1,markersize=4)#,markevery=3)

    ax1.legend(fontsize=35,loc='upper center', bbox_to_anchor=(0.5,1.15), ncols=5, fancybox=True, shadow=True)

    #plt.title("Behaviour SIR model",fontsize=30)
    ax1.tick_params(axis='both', which='major')
    ax1.set_xlabel(r"$t$")


    #ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))

    initial_sum = np.sum(mat[:,0])

    ax2.semilogy(time,np.abs(initial_sum-np.sum(mat,axis=0)),'ko',label=r"$|\delta \mathcal{E}(t)|$ constrained",markersize=10)
    ax2.semilogy(time,np.abs(initial_sum-np.sum(matU,axis=0)),'rd',label=r"$|\delta \mathcal{E}(t)|$ unconstrained",markersize=10)
    ax2.legend(fontsize=35,loc='center', ncol=4, fancybox=True, shadow=True)
    ax2.set_xlabel(r"$t$")

    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))

    # Automatically adjust the layout
    plt.tight_layout()
    plt.savefig("savedPlots/plotSIR_PINN.pdf",bbox_inches='tight')

    #plt.show()