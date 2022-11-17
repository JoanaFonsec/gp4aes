#!/usr/bin/env python3

import numpy as np
import sklearn.gaussian_process as gp
import matplotlib.pyplot as plt
import h5py as h5
import multiprocessing as mp

from pyDOE import lhs

from scipy.interpolate import RegularGridInterpolator

from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from argparse import ArgumentParser

class GroundTruth:
    def __init__(self, chl, lat, lon, time):
        self.chl = chl
        self.lat = lat
        self.lon = lon
        self.time = time


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the HDF5 file containing the processed data.")
    parser.add_argument("timestamp", type=int, help="Time instant where to train/predict [UNIX].")
    parser.add_argument("kernel", type=str, help="RQ for Rational Quadratic kernel or MAT for Matérn one.")
    parser.add_argument("n_days", type=int, help="Number of days of training data to consider, previously to the test day.")
    parser.add_argument("--offset", type=int, default=1, help="Closests day offset.")
    parser.add_argument("--predict", action='store_true', help="Predict CHL density.")
    parser.add_argument("--clipped_op_area", type=float, nargs=4, help="If map contains NaN's, insert smaller operational area in the following order \
                                                                            LAT(min) LAT(max) LON(min) LON(max). Insert 1000 if some value is as default.")

    return parser.parse_args()


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

    def nmlk_multi_dataset(theta):
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

    return nmlk_multi_dataset


def minimize_parallel(args):
    return minimize(nmlk_func, args, bounds=((1e-7, None), (1e-7, None), (1e-7, None)), options={"maxiter":1e5}, method='L-BFGS-B')


def main(args):
    with h5.File(args.path, "r") as f:
        gt = GroundTruth(f["chl"][()], f["lat"][()], f["lon"][()], f["time"][()])

    t_idx = np.argmin(np.abs(args.timestamp - gt.time))

    N = 1000 # train data
    N_meas = 500 # validation data
    n = 200 # test data
    s = 1e-3 # Approximately zero

    # Clip area of operation
    lon_idxs = [0, len(gt.lon)-1]
    lat_idxs = [0, len(gt.lat)-1]

    if args.clipped_op_area:
        lon_idxs = [np.argmin(np.abs(args.clipped_op_area[2]-gt.lon)), np.argmin(np.abs(args.clipped_op_area[3]-gt.lon))]
        lat_idxs = [np.argmin(np.abs(args.clipped_op_area[0]-gt.lat)), np.argmin(np.abs(args.clipped_op_area[1]-gt.lat))]

    # Measurements
    meas_lhd = lhs(2, N_meas)

    X_lon_idxs = np.array((lon_idxs[1] - 1)*meas_lhd[:, 0] + lon_idxs[0], dtype=int).squeeze()
    X_lat_idxs = np.array((lat_idxs[1] - 1)*meas_lhd[:, 1] + lat_idxs[0], dtype=int).squeeze()
    X_idxs = np.vstack((X_lon_idxs, X_lat_idxs)).T

    X = np.array([gt.lon[X_idxs[:,0]], gt.lat[X_idxs[:, 1]]]).T
    y = gt.chl[X_idxs[:,0], X_idxs[:,1], t_idx]

    # Test data
    x = np.zeros((2, n))
    x[0] = np.linspace(gt.lon[0], gt.lon[-1], n)
    x[1] = np.linspace(gt.lat[0], gt.lat[-1], n)

    # Choose kernel
    if args.kernel == 'RQ':
        kernel = gp.kernels.ConstantKernel()*gp.kernels.RationalQuadratic()
    elif args.kernel == 'MAT':
        kernel = gp.kernels.ConstantKernel()*gp.kernels.Matern(length_scale=[1,1])
    else:
        raise ValueError("Invalid kernel. Choices are RQ or MAT.")

    # Init optimization
    f_min = 1e7
    params = [0, 0, 0]

    # (Prior) Training data (Latin-Hypercube)
    X_idxs_per_set = np.ndarray((args.n_days, N, 2), dtype=int)
    X_per_set = np.ndarray((args.n_days, N, 2), dtype=float)
    y_per_set = np.ndarray((args.n_days, N), dtype=float)

    for i in range(args.n_days):
        lhd = lhs(2,N)

        X_lon_idxs = np.array((lon_idxs[1] - 1)*lhd[:, 0] + lon_idxs[0], dtype=int).squeeze()
        X_lat_idxs = np.array((lat_idxs[1] - 1)*lhd[:, 1] + lat_idxs[0], dtype=int).squeeze()

        X_idxs_per_set[i] = np.vstack((X_lon_idxs, X_lat_idxs)).T

        X_per_set[i] = np.array([gt.lon[X_idxs_per_set[i,:,0]], gt.lat[X_idxs_per_set[i,:,1]]]).T
        y_per_set[i] = gt.chl[X_idxs_per_set[i,:,0], X_idxs_per_set[i,:,1], t_idx-args.offset-i]

    # Implement minimization of sum of negative log marginal likelihood of different datasets
    global nmlk_func
    nmlk_func = neg_mlk(X_per_set, y_per_set, s, kernel, args.n_days, kernel_name=args.kernel)

    print("Minimizing...")
    pool = mp.Pool(processes=3)

    if args.kernel == 'RQ':
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

    if args.kernel == 'RQ':
        kernel.set_params(**{'k1__constant_value': params[0], 'k2__length_scale': params[1], 'k2__alpha': params[2]})
    elif args.kernel == 'MAT':
        kernel.set_params(**{'k1__constant_value': params[0], 'k2__length_scale': params[1:]})

    print("Fitting...")
    model = gp.GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=s**2)
    model.fit(X, y)

    lkl = model.log_marginal_likelihood()
    print("Log marginal-likelihood of kernel parameters for training data:", lkl)

    if args.predict:
        # Test predicitons
        xx, yy = np.meshgrid(x[0], x[1])
        x_stack = np.vstack([xx.ravel(), yy.ravel()]).T
        y_pred, std_pred = model.predict(x_stack, return_std=True)

        test_data_vals = RegularGridInterpolator((gt.lon, gt.lat), gt.chl[:,:,t_idx])(x_stack)

        y_pred[np.isnan(test_data_vals)] = np.nan
        rel_error = np.abs(y_pred-test_data_vals)*100/test_data_vals

        # KPIs without NaNs
        y_pred_clean = y_pred[~np.isnan(test_data_vals)]
        test_data_vals_clean = test_data_vals[~np.isnan(test_data_vals)]
        av_rel_error = np.mean(np.abs(y_pred_clean-test_data_vals_clean)*100/test_data_vals_clean)
        av_std = np.mean(std_pred[~np.isnan(test_data_vals)])

        # Relative error
        print("Average relative error: %.3f%%" % np.mean(av_rel_error))
        print("Average std: %.3f" % av_std)

        # Write data on output file
        with h5.File(args.path, "w") as f:
            f.create_dataset("chl", data=gt.chl)
            f.create_dataset("lat", data=gt.lat)
            f.create_dataset("lon", data=gt.lon)
            f.create_dataset("time", data=gt.time)
            f.create_dataset("X", data=X)
            f.create_dataset("y", data=y)
            f.create_dataset("x", data=x)
            f.create_dataset("y_pred", data=y_pred)
            f.create_dataset("std_pred", data=std_pred)
            f.create_dataset("rel_error", data=rel_error)
            f.attrs.create("av_rel_error", data=av_rel_error)
            f.attrs.create("t_idx", data=t_idx)


if __name__ == "__main__":
    args = parse_args()
    main(args)