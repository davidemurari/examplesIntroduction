import torch
from tqdm import tqdm
from torch.func import vmap, jacfwd
from scripts.plotting import testPlot
from scripts.utils import *
from scripts.plotting import *

def trainModel(model,n_train,q0,pi0,dt,bounds,vec,epochs,device,dtype,optimizer,verbose=0):

    #verbose = 0 : No value displayed
    #verbose = 1 : Loss values displayed every few epochs
    #verbose = 2 : Loss values and plots every few epochs
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=int(0.4*epochs),gamma=0.1)
    
    t0,tf,qlb,qub,pilb,piub = bounds
    d = model.d
    best = 1.
        
    for epoch in tqdm(range(epochs)):
        
        q = torch.rand((n_train,d//2),dtype=dtype)*(qub-qlb)+qlb
        pi = torch.rand((n_train,d//2),dtype=dtype)*(piub-pilb)+pilb
        t = t0 + dt * torch.rand((n_train,1),dtype=dtype).to(device)
        z = torch.cat((q,pi),dim=1).to(device)
        
        optimizer.zero_grad()    
                            
        z0 = z.clone()
    
        z = model(z0,t)
                            
        derivative = lambda x,t : vmap(jacfwd(model,argnums=1))(x,t)
        z_d = derivative(z0,t).squeeze()                
        LossResidual = vec.residualLoss(z,z_d,t)
        
        Ltot = LossResidual
                    
        Ltot.backward(retain_graph=False);
        optimizer.step();
        scheduler.step();
        
        if epoch%500==0:
            if verbose==1 or verbose==2:
                print(f"Epoch {epoch}, Error Residual {LossResidual.item()}")
            if verbose==2:
                testPlot(vec,q0,pi0,tf,model,dtype,device) #Plots the solution for one initial conditio to see how it progresses while training

        if epoch>0.8*epochs:
            if Ltot.item()<best:
                path = f"trainedModels/best.pt"
                torch.save(model.state_dict(), path)
                best = Ltot.item()
            
    return Ltot.item()