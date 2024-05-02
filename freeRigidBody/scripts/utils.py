import numpy as np
import scipy

def hat(v):
    A = np.zeros((3,3))
    A[0,1] = -v[2]
    A[0,2] = v[1]
    A[1,2] = -v[0]

    A = A - A.T
    return A

def expSO3(w):
    A = hat(w)
    eA = scipy.linalg.expm(A)
    return eA

H = lambda y : (y[0]**2+y[1]**2+y[2]**2)
def B(y):
    I1,I2,I3 = 2.,1.,2/3
    return np.array([[0.,y[2]/I3,-y[1]/I2],[-y[2]/I3,0.,y[0]/I1],[y[1]/I2,-y[0]/I1,0.]])

f = lambda y : B(y)@y

def df(y):
    I1,I2,I3 = 2.,1.,2/3
    return np.array([
    [0,y[2]/I3-y[2]/I3,y[1]/I3-y[1]/I2],
    [-y[2]/I3+y[2]/I1,0.,-y[0]/I3+y[0]/I1],
    [y[1]/I2-y[1]/I1,y[0]/I2-y[0]/I1,0.]
])

def fManiAlgebra(y):
    I1,I2,I3 = 2.,1.,2/3
    return np.array([-y[0]/I1,-y[1]/I2,-y[2]/I3])

lieEulerStep = lambda h,y: expSO3(h*fManiAlgebra(y))@y