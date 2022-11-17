#!/usr/bin/env python3

import numpy as np
import sklearn.gaussian_process as gp
import h5py as h5
from pyDOE import lhs

from scipy.interpolate import RegularGridInterpolator

from argparse import ArgumentParser

import time


class GridData:
    def __init__(self, chl, lat, lon, time):
        self.chl = chl
        self.lat = lat
        self.lon = lon
        self.time = time


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the HDF5 file containing the processed training data.")
    parser.add_argument("traj", type=str, help="Path to the HDF5 containing the trajectory training data")
    parser.add_argument("timestamp", type=int, help="Time instant where to train/predict [UNIX].")
    parser.add_argument("kernel", type=str, help="RQ for Rational Quadratic kernel or MAT for Matérn one.")
    parser.add_argument("--kernel_params", nargs=3, type=float, help="3 kernel parameters.")
    parser.add_argument("--clipped_op_area", type=float, nargs=4, help="Insert smaller operational area in the following order \
                                                                            LAT(min) LAT(max) LON(min) LON(max). Insert 1000 if some value is as default.")
    parser.add_argument("--use_curr_X", action='store_true', help="Use existing measurement dataset.")
    parser.add_argument("--cont_std", action='store_true', help='Consider std of scattered observations as a function of the distance to the trajectory.')
    return parser.parse_args()


def main(args):
    with h5.File(args.path, "r") as f:
        data = GridData(f["chl"][()], f["lat"][()], f["lon"][()], f["time"][()])

    with h5.File(args.traj, "r") as f:
        meas_per = f.attrs["meas_per"]
        t_idx = f.attrs["t_idx"]
        X_traj = f["traj"][0:-1:meas_per,:]
        y_traj = f["delta_vals"][0:-1]

    N_traj = 1300 # Number of trajectory points
    var_traj = 1e-10
    N_meas = 400 # number of measurements points
    n = int(np.sqrt(9*(N_meas+N_traj))*1.1) # number of test points per dimension (10/90 ratio)
    s = 1e-1 # noise standard deviation

    # Training area
    if args.use_curr_X:
        with h5.File(args.path, "r") as f:
            X = f["X"][()]
            y = f["y"][()]
    else:
        # Clip area of operation
        lon_idxs = [0, len(data.lon)-1]
        lat_idxs = [0, len(data.lat)-1]

        if args.clipped_op_area:
            lon_idxs = [np.argmin(np.abs(args.clipped_op_area[2]-data.lon)), np.argmin(np.abs(args.clipped_op_area[3]-data.lon))]
            lat_idxs = [np.argmin(np.abs(args.clipped_op_area[0]-data.lat)), np.argmin(np.abs(args.clipped_op_area[1]-data.lat))]

        # Latin-Hypercube
        lhd = lhs(2, N_meas)

        X_lon_idxs = np.array((lon_idxs[1] - 1)*lhd[:, 0] + lon_idxs[0], dtype=int).squeeze()
        X_lat_idxs = np.array((lat_idxs[1] - 1)*lhd[:, 1] + lat_idxs[0], dtype=int).squeeze()

        X_idxs = np.vstack((X_lon_idxs, X_lat_idxs)).T
        X = np.array([data.lon[X_idxs[:, 0]], data.lat[X_idxs[:, 1]]]).T

        y = data.chl[X_idxs[:,0], X_idxs[:,1], t_idx] + np.random.normal(0, s, size=X_idxs.shape[0])

    if np.count_nonzero(np.isnan(y)) > 0:
        # Udate n to mantain ratio between train and test dataset size
        n = int(np.sqrt(9*(N_meas+N_traj - np.count_nonzero(np.isnan(y)))) * 1.1)
        X = X[~np.isnan(y), :]
        y = y[~np.isnan(y)]

    # Test data
    x = np.zeros((2, n))
    x[0] = np.linspace(data.lon[0], data.lon[-1], n)
    x[1] = np.linspace(data.lat[0], data.lat[-1], n)

    xx, yy = np.meshgrid(x[0], x[1])
    x_stack = np.vstack([xx.ravel(), yy.ravel()]).T

    test_data_vals = RegularGridInterpolator((data.lon, data.lat), data.chl[:,:,t_idx])(x_stack)

    # Choose kernel and define its parameters
    if not args.kernel_params:
        if args.kernel == 'RQ':
            kernel = gp.kernels.ConstantKernel()*gp.kernels.RationalQuadratic(length_scale=1, alpha=1e-3)
        elif args.kernel == 'MAT':
            kernel = gp.kernels.ConstantKernel()*gp.kernels.Matern(length_scale=[1, 1])
        else:
            raise ValueError("Invalid kernel. Chooose either RQ or MAT.")

    else:
        if args.kernel == 'RQ':
            kernel = gp.kernels.ConstantKernel(args.kernel_params[0])*gp.kernels.RationalQuadratic(length_scale=args.kernel_params[1], alpha=args.kernel_params[2])
        elif args.kernel == 'MAT':
            kernel = gp.kernels.ConstantKernel(args.kernel_params[0])*gp.kernels.Matern(length_scale=[args.kernel_params[1], args.kernel_params[2]])
        else:
            raise ValueError("Invalid kernel. Chooose either RQ or MAT.")

    print("Kernel hyperparameters:", kernel.get_params(deep=False))

    # Concatenate scattered data and traj data
    traj_step = int(X_traj.shape[0]/N_traj)
    X_traj = X_traj[0:-1:traj_step]
    y_traj = y_traj[0:-1:traj_step]

    print("N_train =", X.shape[0], "N_traj = ", X_traj.shape, " n =", n**2)

    if X_traj.shape[0] == y_traj.shape[0] + 1:
        X_traj = X_traj[:-1]

    X_meas = np.vstack((X, X_traj))
    y_meas = np.hstack((y, y_traj)).T


    if not args.cont_std:
        alpha_meas = np.vstack((np.ones((X.shape[0], 1))*(s**2), np.ones((X_traj.shape[0], 1))*(var_traj)))
    else:
        # std = f(dist to traj)
        max_std = 0.5
        min_std = 0.05

        # linear params
        a = (max_std - min_std) / (-max([np.linalg.norm(x_s - x_t) for x_s in X for x_t in X_traj]))
        b = min_std

        alpha_meas = np.zeros((X.shape[0]+X_traj.shape[0], 1))

        for i in range(X.shape[0]-1):
            alpha_meas[i] = (a*min([np.linalg.norm(X[i, :] - x_t) for x_t in X_traj]) + b)**2

        alpha_meas[X.shape[0]:] = var_traj

    # Fit GP model with measurements
    print("Fitting GP model...")
    start = time.time()
    model = gp.GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=alpha_meas.squeeze())
    model.fit(X_meas, y_meas)
    print("Time taken for fitting:", time.time() - start)

    # Test data
    print("Predicting...")
    y_pred, std_pred = model.predict(x_stack, return_std=True)
    y_pred[np.isnan(test_data_vals)] = np.nan
    rel_error = np.abs(y_pred-test_data_vals)*100/test_data_vals

    # KPIs without NaNs
    y_pred_clean = y_pred[~np.isnan(test_data_vals)]
    test_data_vals_clean = test_data_vals[~np.isnan(test_data_vals)]
    av_rel_error = np.mean(np.abs(y_pred_clean-test_data_vals_clean)*100/test_data_vals_clean)
    av_std = np.mean(std_pred[~np.isnan(test_data_vals)])

    print("Average relative error: %.3f%%" % av_rel_error)
    print("Average std: %.3f" % av_std)

    # Write data on output file
    with h5.File(args.path, "w") as f:
        f.create_dataset("chl", data=data.chl)
        f.create_dataset("lat", data=data.lat)
        f.create_dataset("lon", data=data.lon)
        f.create_dataset("time", data=data.time)
        f.create_dataset("X", data=X_meas)
        f.create_dataset("y", data=y_meas)
        f.create_dataset("x", data=x)
        f.create_dataset("y_pred", data=y_pred)
        f.create_dataset("std_pred", data=std_pred)
        f.create_dataset("rel_error", data=rel_error)
        f.attrs.create("av_rel_error", data=av_rel_error)
        f.attrs.create("t_idx", data=t_idx)


if __name__ == "__main__":
    args = parse_args()
    main(args)