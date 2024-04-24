import numpy as np
import scipy.integrate


def generate(N, M, T, Ntrain, noisy=False):
    # N : how many points to generate
    # M : number of time steps (including the initial condition)
    # T : final time
    # Ntrain : percentage of points in the training

    np.random.seed(0)

    dim = 3

    R0 = 1
    f = lambda t, y: np.array([-R0 * y[0] * y[1], R0 * y[0] * y[1] - y[1], y[1]])

    X = np.random.rand(N, dim)
    X = X #/ np.sum(X, axis=1).reshape(-1, 1)
    NN = int(Ntrain * N)

    time = np.linspace(0, T, M)
    h = time[1] - time[0]
    traj = np.zeros([N, dim, M])
    for i in range(N):
        traj[i, :, :] = scipy.integrate.solve_ivp(
            f, [0, T], X[i], method="RK45", t_eval=time).y

    Xtrain, ytrain = traj[:NN, :, 0], traj[:NN, :, 1:]
    Xtest, ytest = traj[NN:, :, 0], traj[NN:, :, 1:]

    if noisy:
        ytrain += np.random.rand(*ytrain.shape) * 0.01  # noisy training trajectories

    return Xtrain, ytrain, Xtest, ytest, h
