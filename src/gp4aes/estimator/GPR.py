import numpy as np
from scipy.spatial.distance import cdist
import sklearn.gaussian_process as gp
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
import multiprocessing as mp
from pyDOE import lhs

class GPEstimator:
    def __init__(self, kernel, s, range_m, params=None, earth_radius=6369345):
        if not (kernel == 'RQ' or kernel == 'MAT'):
            raise ValueError("Invalid kernel. Choices are RQ or MAT.")

        if params is not None:
            if kernel == 'RQ':
                self.__kernel = gp.kernels.ConstantKernel(params[0])*gp.kernels.RationalQuadratic(length_scale=params[2], alpha=params[3])

            elif kernel == 'MAT':
                self.__kernel = gp.kernels.ConstantKernel(params[0])*gp.kernels.Matern(length_scale=params[1:])

        else:
            if kernel == 'RQ':
                self.__kernel = gp.kernels.ConstantKernel(91.2025)*gp.kernels.RationalQuadratic(length_scale=0.00503, alpha=0.0717)
            elif kernel == 'MAT':
                self.__kernel = gp.kernels.ConstantKernel(44.29588721)*gp.kernels.Matern(length_scale=[0.54654887, 0.26656638])

        self.__kernel_name = kernel

        self.s = s
        self.__model = gp.GaussianProcessRegressor(kernel=self.__kernel, optimizer=None, alpha=self.s**2)

        # Estimation range where to predict values
        self.range_deg = range_m / (np.radians(1.0) * earth_radius)


    """
    Gaussian Process Regression - Gradient analytical estimation

    Parameters
    ----------
    X: trajectory coordinates array
    y: Measurements on X coordinates
    dist_metric: distance metric used to calculate distances
    """
    def est_grad(self, X, y, dist_metric='euclidean'):
        self.__model.fit(X[:-1], y[:-1])
        x = np.atleast_2d(X[-1])

        params = self.__kernel.get_params()

        if self.__kernel_name == 'RQ':
            sigma = params["k1__constant_value"]
            length_scale = params["k2__length_scale"]
            alpha = params["k2__alpha"]

            dists = cdist(x, X[:-1], metric=dist_metric)
            x_dist = nonabs_1D_dist(x[:,0], X[:-1,0])
            y_dist = nonabs_1D_dist(x[:,1], X[:-1,1])

            common_term = 1 + dists ** 2 / (2 * alpha * length_scale ** 2)
            common_term = common_term ** (-alpha-1)
            common_term = -sigma /  (length_scale ** 2) * common_term

            dx = x_dist * common_term
            dy = y_dist * common_term

        elif self.__kernel_name == 'MAT':
            sigma = params["k1__constant_value"]
            length_scale = params["k2__length_scale"]

            dists = cdist(x/length_scale, X[:-1]/length_scale, metric=dist_metric)

            dists = dists * np.sqrt(3)

            x_dist = nonabs_1D_dist(x[:,0], X[:-1,0]) / (length_scale[0]**2)
            y_dist = nonabs_1D_dist(x[:,1], X[:-1,1]) / (length_scale[1]**2)

            common_term = -3 * sigma * np.exp(-dists)

            dx = x_dist * common_term
            dy = y_dist * common_term

        return dx @ self.__model.alpha_, dy @ self.__model.alpha_


def nonabs_1D_dist(x, X):
    res = np.zeros((x.shape[0], X.shape[0]))

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = x[i] - X[j]

    return res

"""
    Negative Log-Marginal Likelihood
    Parameters
    ----------
    X: 2D coordinate array of the [concatenated] test locations
    y: Measurements on X coordinates
    s: Noise standard deviation
    kernel: Covariance function (prior function to be learned)
    n: Number of sub datasets
    Return
    ------
    Returns a function that computes the sum of the negative log marginal
    likelihood for training data X and Y, given some noise level. The training
    dataset starts on n_before days before the test day and finishes n_after
    days after the test day.
"""
def neg_mlk(X_per_dataset, y_per_dataset, s, kernel, n, kernel_name='RQ'):
    if not (kernel_name == 'RQ' or kernel_name == 'MAT'):
        raise ValueError("Invalid kernel.")

    def nmlk_func(theta):
        val = 0

        for i in range(n):
            if kernel_name == 'RQ':
                kernel.set_params(**{'k1__constant_value': theta[0], 'k2__length_scale': theta[1], 'k2__alpha': theta[2]})
                K = kernel(X_per_dataset[i]) + \
                    s**2 * np.eye(len(X_per_dataset[i]))
            elif kernel_name == 'MAT':
                # Anisotropic -> theta must be 2D
                kernel.set_params(**{'k1__constant_value': theta[0], 'k2__length_scale': theta[1:]})
                K = kernel(X_per_dataset[i]) + \
                    s**2 * np.eye(len(X_per_dataset[i]))

            L = cholesky(K)

            S1 = solve_triangular(L, y_per_dataset[i], lower=True)
            S2 = solve_triangular(L.T, S1, lower=False)

            val = val + np.sum(np.log(np.diagonal(L))) + \
                        0.5 * y_per_dataset[i].dot(S2) + \
                        0.5 * len(y_per_dataset[i]) * np.log(2*np.pi)

        return val

    return nmlk_func


def minimize_parallel(args):
    return minimize(nmlk_func, args, bounds=((1e-7, None), (1e-7, None), (1e-7, None)), options={"maxiter":1e5}, method='L-BFGS-B')

def train_GP_model(chl, lat, lon, s, N, N_meas, n, n_days, t_idx, offset, clipped_area, kernel_name):

    # Clip area of operation
    lon_idxs = [0, len(lon)-1]
    lat_idxs = [0, len(lat)-1]

    if clipped_area is not None:
        lon_idxs = [np.argmin(np.abs(clipped_area[2]-lon)), np.argmin(np.abs(clipped_area[3]-lon))]
        lat_idxs = [np.argmin(np.abs(clipped_area[0]-lat)), np.argmin(np.abs(clipped_area[1]-lat))]

    # Measurements
    meas_lhd = lhs(2, N_meas)

    X_lon_idxs = np.array((lon_idxs[1] - 1)*meas_lhd[:, 0] + lon_idxs[0], dtype=int).squeeze()
    X_lat_idxs = np.array((lat_idxs[1] - 1)*meas_lhd[:, 1] + lat_idxs[0], dtype=int).squeeze()
    X_idxs = np.vstack((X_lon_idxs, X_lat_idxs)).T

    X = np.array([lon[X_idxs[:,0]], lat[X_idxs[:, 1]]]).T
    y = chl[X_idxs[:,0], X_idxs[:,1], t_idx]

    # Test data
    x = np.zeros((2, n))
    x[0] = np.linspace(lon[0], lon[-1], n)
    x[1] = np.linspace(lat[0], lat[-1], n)

    # Choose kernel
    if kernel_name == 'RQ':
        kernel = gp.kernels.ConstantKernel()*gp.kernels.RationalQuadratic()
    elif kernel_name == 'MAT':
        kernel = gp.kernels.ConstantKernel()*gp.kernels.Matern(length_scale=[1,1])
    else:
        raise ValueError("Invalid kernel. Choices are RQ or MAT.")

    # Init optimization
    f_min = 1e7
    params = [0, 0, 0]

    # (Prior) Training data (Latin-Hypercube)
    X_idxs_per_set = np.ndarray((n_days, N, 2), dtype=int)
    X_per_set = np.ndarray((n_days, N, 2), dtype=float)
    y_per_set = np.ndarray((n_days, N), dtype=float)

    for i in range(n_days):
        lhd = lhs(2,N)

        X_lon_idxs = np.array((lon_idxs[1] - 1)*lhd[:, 0] + lon_idxs[0], dtype=int).squeeze()
        X_lat_idxs = np.array((lat_idxs[1] - 1)*lhd[:, 1] + lat_idxs[0], dtype=int).squeeze()

        X_idxs_per_set[i] = np.vstack((X_lon_idxs, X_lat_idxs)).T

        X_per_set[i] = np.array([lon[X_idxs_per_set[i,:,0]], lat[X_idxs_per_set[i,:,1]]]).T
        y_per_set[i] = chl[X_idxs_per_set[i,:,0], X_idxs_per_set[i,:,1], t_idx-offset-i]

    # Implement minimization of sum of negative log marginal likelihood of different datasets
    global nmlk_func
    nmlk_func = neg_mlk(X_per_set, y_per_set, s, kernel, n_days, kernel_name)

    print("Minimizing...")
    pool = mp.Pool(processes=3)

    if kernel_name == 'RQ':
        init = np.vstack((lhs(1,4).squeeze()*2, lhs(1,4).squeeze()*4, lhs(1,4).squeeze()*0.01)).T
    else:
        init = np.vstack((lhs(1,10).squeeze()*50, lhs(1,10).squeeze(), lhs(1,10).squeeze())).T
        # init = [[27.76435108, 0.46550372, 0.22160483]]

    results = pool.map(minimize_parallel, init)

    for res in results:
        if res["fun"] < f_min:
            f_min = res["fun"]
            params = res["x"]

    print("Parameters derived from MLE:", params)

    if kernel_name == 'RQ':
        kernel.set_params(**{'k1__constant_value': params[0], 'k2__length_scale': params[1], 'k2__alpha': params[2]})
    elif kernel_name == 'MAT':
        kernel.set_params(**{'k1__constant_value': params[0], 'k2__length_scale': params[1:]})

    print("Fitting...")
    model = gp.GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=s**2)
    model.fit(X, y)

    lkl = model.log_marginal_likelihood()
    print("Log marginal-likelihood of kernel parameters for training data:", lkl)

    return params