
import torch
import numpy as np
import scipy

def flow(y,model,dtype):
    if dtype==torch.float64:
        z = torch.from_numpy(y.astype(np.float64)).reshape(1,-1)
    else:
        z = torch.from_numpy(y.astype(np.float32)).reshape(1,-1)
    return model(z).detach().numpy().reshape(-1)

def get_solutions(it,y0,net,netU,h,dtype):
    
    R0 = 1
    f = lambda t,y: np.array([-R0*y[0]*y[1],
                                R0*y[0]*y[1] - y[1],
                                y[1]
        ])
    
    flowC = lambda y : flow(y,net,dtype)
    flowU = lambda y : flow(y,netU,dtype)

    point = y0.copy()
    pointU = y0.copy()
    tt = 0
    mat = np.zeros((3,it))
    mat[:,0] = point
    matU = np.zeros((3,it))
    matU[:,0] = pointU
    time = [0]
    for c in range(it-1):
        tt+=h
        time.append(tt)
        point = flowC(point)
        pointU = flowU(pointU)
        mat[:,c+1] = point
        matU[:,c+1] = pointU
    sol_true = scipy.integrate.solve_ivp(f,[0,max(time)],y0,method='RK45',t_eval=time,atol=1e-11,rtol=1e-11).y

    return time,sol_true,mat,matU