import numpy as np
import math 
from scipy.spatial.distance import cdist
import sklearn.gaussian_process as gp

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


def grad_moving_average_2D(grad, idx, n, weights):
    val_x = grad[idx - n+1:idx + 1, 0]
    val_y = grad[idx - n+1:idx + 1, 1]

    x_ma = np.average(val_x, weights=weights)
    y_ma = np.average(val_y, weights=weights)

    return np.array([x_ma, y_ma])

