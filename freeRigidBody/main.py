import numpy as np
import scipy
import os

from scripts.utils import *
from scripts.plotting import *

import os

os.chdir("./freeRigidBody")

I1,I2,I3 = 2.,1.,2/3
y0 = np.array([np.cos(1.1),0.,np.sin(1.1)])

h = 0.05
N = 800

sol_EE = np.zeros((3,N+1))
sol_LE = np.zeros((3,N+1))

sol_EE[:,0] = y0
sol_LE[:,0] = y0

for i in range(N):
  sol_EE[:,i+1] = sol_EE[:,i] + h * f(sol_EE[:,i])

for i in range(N):
  sol_LE[:,i+1] = lieEulerStep(h,sol_LE[:,i])
  
plot(y0,N,h,sol_LE,sol_EE)