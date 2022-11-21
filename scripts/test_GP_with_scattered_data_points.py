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
    parser.add_argument("timestamp", type=int, help="Time instant where to train/predict [UNIX].")
    parser.add_argument("kernel", type=str, help="RQ for Rational Quadratic kernel or MAT for Matérn one.")
    parser.add_argument("--kernel_params", nargs=3, type=float, help="3 kernel parameters.")
    parser.add_argument("--train_line", nargs=5, type=float, help="Line training data instead of scattered. Insert line end points coordinates in (Lon, Lat) format.")
    parser.add_argument("--train_traj", type=str, help="Path to the HDF5 containing the trajectory training data")
    parser.add_argument("--test_line", nargs=5, type=float, help="Line test data instead of scattered. Insert line end points coordinates in (Lon, Lat) format. \
                                                                Add range [m] in the end or 0 if test data should be the entire area.")
    parser.add_argument("--test_path", type=str, help="Path to the HDF5 file containing the processed test data.")
    parser.add_argument("--clipped_op_area", type=float, nargs=4, help="Insert smaller operational area in the following order \
                                                                            LAT(min) LAT(max) LON(min) LON(max). Insert 1000 if some value is as default.")
    parser.add_argument("--use_curr_X", action='store_true', help="Use existing measurement dataset.")

    return parser.parse_args()


def main(args):
    with h5.File(args.path, "r") as f:
        gt = GridData(f["chl"][()], f["lat"][()], f["lon"][()], f["time"][()])

    if args.test_path:
        with h5.File(args.test_path, "r") as f:
            data = GridData(f["chl"][()], f["lat"][()], f["lon"][()], f["time"][()])
    else:
        data = gt

    t_idx = np.argmin(np.abs(args.timestamp - data.time))

    N = 1000 #1500 # number of training points
    N_meas = 500 #int(1700*1.1) # number of measurements points
    n = 200 #int(np.sqrt(9*N_meas)*1.1) # number of test points per dimension (10/90 ratio)
    s = 1e-3 # noise standard deviation

    # Kernel-training data
    if not args.kernel_params:
        # Clip area of operation
        lon_idxs = [0, len(gt.lon)-1]
        lat_idxs = [0, len(gt.lat)-1]

        if args.clipped_op_area:
            lon_idxs = [np.argmin(np.abs(args.clipped_op_area[2]-gt.lon)), np.argmin(np.abs(args.clipped_op_area[3]-gt.lon))]
            lat_idxs = [np.argmin(np.abs(args.clipped_op_area[0]-gt.lat)), np.argmin(np.abs(args.clipped_op_area[1]-gt.lat))]

        # Latin-Hypercube
        lhd = lhs(2, N)

        X_lon_idxs = np.array((len(gt.lon) - 1)*lhd[:, 0], dtype=int).squeeze()
        X_lat_idxs = np.array((len(gt.lat) - 1)*lhd[:, 1], dtype=int).squeeze()

        X_idxs = np.vstack((X_lon_idxs, X_lat_idxs)).T
        X_train = np.array([gt.lon[X_idxs[:, 0]], gt.lat[X_idxs[:, 1]]]).T

        y_train = gt.chl[X_idxs[:,0], X_idxs[:,1], t_idx] + np.random.normal(0, s, size=X_idxs.shape[0])

        X_train = X_train[~np.isnan(y_train), :]
        y_train = y_train[~np.isnan(y_train)]

    # Training data
    if args.use_curr_X:
        with h5.File(args.path if not args.test_path else args.test_path, "r") as f:
            X = f["X"][()]
            y = f["y"][()]
            n = f["x"].shape[1]

    else:
        if args.train_line:
            x0 = (args.train_line[0], args.train_line[1])
            x1 = (args.train_line[2], args.train_line[3])

            if x0[0] == x1[0]:
                lon_coords = np.tile(x0[0], N_meas)
                lat_coords = np.linspace(x0[1], x1[1], N_meas)

                X = np.vstack((lon_coords, lat_coords)).T

                idxs_lat = np.zeros(N_meas, dtype=int)
                idx_lon = np.argmin(np.abs(x0[0] - data.lon))
                for i in range(X.shape[0]):
                    idxs_lat[i] = np.argmin(np.abs(X[i,1] - data.lat))

                y = data.chl[idx_lon, idxs_lat[:], t_idx] + np.random.normal(0, s, size=N)

            elif x0[1] == x1[1]:
                lon_coords = np.linspace(x0[0], x1[0], N_meas)
                lat_coords = np.tile(x0[1], N_meas)

                X = np.vstack((lon_coords, lat_coords)).T

                idx_lat = np.argmin(np.abs(x0[1] - data.lat))
                idxs_lon = np.zeros(N_meas, dtype=int)
                for i in range(X.shape[0]):
                    idxs_lon[i] = np.argmin(np.abs(X[i,0] - data.lon))

                y = data.chl[idxs_lon[:], idx_lat, t_idx] + np.random.normal(0, s, size=N_meas)

            else:
                slope = (x1[1] - x0[1]) / (x1[0] - x0[0])
                intercept = (x0[0]*x1[1] - x1[0]*x0[1])/(x0[0]-x1[0])

                lon_coords = np.linspace(x0[0], x1[0], N_meas)

                lat_coords = slope*lon_coords + intercept

                X = np.vstack((lon_coords, lat_coords)).T

                idxs = np.zeros(X.shape, dtype=int)
                for i in range(X.shape[0]):
                    idxs[i, 0] = np.argmin(np.abs(X[i,0] - data.lon))
                    idxs[i, 1] = np.argmin(np.abs(X[i,1] - data.lat))

                y = data.chl[idxs[:, 0], idxs[:, 1], t_idx] + np.random.normal(0, s, size=N_meas)

        elif args.train_traj:
            with h5.File(args.traj, "r") as f:
                X = f["traj"][0:-1:20,:]
                y = f["delta_vals"][0:-1:20]

                # Choose just part of the trajectory as measurements
                X = X[int(X.shape[0]/10*0):int(X.shape[0]/10*6)]
                y = y[int(y.shape[0]/10*0):int(y.shape[0]/10*6)]

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
                n = int(np.sqrt(9*(N_meas - np.count_nonzero(np.isnan(y))))*1.1)
                X = X[~np.isnan(y), :]
                y = y[~np.isnan(y)]

    # Test data
    if args.test_line and args.test_line[4] > 0:
        range_deg = args.test_line[4] / (np.radians(1.0) * 6369.345 * 1e3)

        if x0[0] == x1[0]:
            x = np.zeros((2,n))
            x[0] = np.linspace(x0[0] - range_deg, x0[0] + range_deg, n)
            x[1] = np.linspace(x0[1], x1[1], n)

        if x0[1] == x1[1]:
            x = np.zeros((2,n))
            x[0] = np.linspace(x0[0], x1[0], n)
            x[1] = np.linspace(x0[1] - range_deg, x0[1] + range_deg, n)

    else:
        x = np.zeros((2, n))
        x[0] = np.linspace(data.lon[0], data.lon[-1], n)
        x[1] = np.linspace(data.lat[0], data.lat[-1], n)

    xx, yy = np.meshgrid(x[0], x[1])
    x_stack = np.vstack([xx.ravel(), yy.ravel()]).T

    test_data_vals = RegularGridInterpolator((data.lon, data.lat), data.chl[:,:,t_idx])(x_stack)

    print(n, np.sqrt(n**2 - test_data_vals[np.isnan(test_data_vals)].shape[0]), n, N_meas)

    # Choose kernel and define its parameters
    if args.kernel == 'RQ':
        kernel = gp.kernels.ConstantKernel()*gp.kernels.RationalQuadratic(length_scale=1, alpha=1e-3)
    elif args.kernel == 'MAT':
        kernel = gp.kernels.ConstantKernel()*gp.kernels.Matern(length_scale=[1, 1])
    else:
        raise ValueError("Invalid kernel. Chooose either RQ or MAT.")

    # Train kernel [if no parameters were given]
    if not args.kernel_params:
        print("Learning kernel hyperparameters...")
        start = time.time()
        restarts = 10
        model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=restarts, alpha=s**2)
        model.fit(X_train, y_train)
        kernel = model.kernel_
        print("Time taken for training:", time.time() - start)

    else:
        if args.kernel == 'RQ':
            kernel.set_params(**{'k1__constant_value': args.kernel_params[0], 'k2__length_scale': args.kernel_params[1], \
                                    'k3__alpha': args.kernel_params[2]})
        elif args.kernel == 'MAT':
            kernel.set_params(**{'k1__constant_value': args.kernel_params[0], 'k2__length_scale': args.kernel_params[1:]})

    print("Kernel hyperparameters:", kernel.get_params(deep=False))
    
    # Fit GP model with measurements
    print("Fitting GP model...")
    print("N_train =", X.shape[0], " n=", n**2)
    start = time.time()
    restarts = 10
    model = gp.GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=s**2)
    model.fit(X, y)
    print("Time taken for fitting:", time.time() - start)

    lmlkh = model.log_marginal_likelihood()
    print("Log marginal-likelihood of kernel parameters:", lmlkh)

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
    with h5.File(args.path if not args.test_path else args.test_path, "w") as f:
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