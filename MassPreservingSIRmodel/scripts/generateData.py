import numpy as np
import scipy.integrate
import torch
from torch.utils.data import Dataset, DataLoader

def generate(N, M, T, Ntrain):
    # N : how many points to generate
    # M : number of time steps (including the initial condition)
    # T : final time
    # Ntrain : percentage of points in the training

    np.random.seed(0)

    dim = 3

    R0 = 1
    f = lambda t, y: np.array([-R0 * y[0] * y[1], R0 * y[0] * y[1] - y[1], y[1]])

    X = np.random.rand(N, dim)
    X = X
    NN = int(Ntrain * N)

    time = np.linspace(0, T, M)
    h = time[1] - time[0]
    traj = np.zeros([N, dim, M])
    for i in range(N):
        traj[i, :, :] = scipy.integrate.solve_ivp(
            f, [0, T], X[i], method="RK45", t_eval=time).y

    Xtrain, ytrain = traj[:NN, :, 0], traj[:NN, :, 1:]
    Xtest, ytest = traj[NN:, :, 0], traj[NN:, :, 1:]
    return Xtrain, ytrain, Xtest, ytest, h

class dataset(Dataset):
  def __init__(self,x,y,dtype):
    if dtype==torch.float64:
      self.x = torch.from_numpy(x.astype(np.float64))
      self.y = torch.from_numpy(y.astype(np.float64))
    else:
      self.x = torch.from_numpy(x.astype(np.float32))
      self.y = torch.from_numpy(y.astype(np.float32))
    self.length = self.x.shape[0]

  def __getitem__(self,idx):
    return self.x[idx], self.y[idx]
  def __len__(self):
    return self.length #num elementi del dataset

def get_data_loaders(x_train,y_train,x_test,y_test,batch_size,dtype=torch.float32):
    trainset = dataset(x_train,y_train,dtype)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = dataset(x_test,y_test,dtype)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    return trainset,testset,trainloader,testloader