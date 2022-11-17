#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
import h5py as h5

from scipy.interpolate import RegularGridInterpolator

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
    parser.add_argument("kernel", type=str, help="RQ for Rational Quadratic kernel or MAT for Matérn one.")
    parser.add_argument("traj", type=str, help="Path to the HDF5 containing the trajectory")
    parser.add_argument("range", type=float, help="Range to evaluate/plot the error in m.")

    return parser.parse_args()


def get_ortogonal_points(X, i, max_dist, n=200):
    if i == 0:
        m_0 = (X[i+1, 1] - X[i, 1]) / (X[i+1, 0] - X[i, 0])
    elif i == X.shape[0]-1:
        m_0 = (X[i, 1] - X[i-1, 1]) / (X[i, 0] - X[i-1, 0])
    else:
        m_0 = (X[i+1, 1] - X[i-1, 1]) / (X[i+1, 0] - X[i-1, 0])

    m_p = -1/m_0
    b_p = X[i, 1] - m_p*X[i, 0]

    theta = np.arctan(m_p)

    x_right = X[i,0] + max_dist*np.cos(theta)
    x_left = X[i,0] - max_dist*np.cos(theta)

    x = np.linspace(x_left, x_right, n)

    line_points = np.vstack((x, m_p*x + b_p)).T

    return line_points


def main(args):
    with h5.File(args.path, "r") as f:
        gt = GroundTruth(f["chl"][()], f["lat"][()], f["lon"][()], f["time"][()])

    s = 5e-3 # noise standard deviation

    with h5.File(args.traj, "r") as f:
        t_idx = f.attrs["t_idx"]
        meas_per = f.attrs["meas_per"]
        X = f["traj"][0:-1:meas_per, :]
        X = X[0:-1:200,:]
        y = f["delta_vals"][0:-1:200]

    # Choose kernel
    if args.kernel == 'RQ':
        kernel = gp.kernels.ConstantKernel(1.47762564)*gp.kernels.RationalQuadratic(length_scale=0.48351041, alpha=0.01123431)
    elif args.kernel == 'MAT':
        kernel = gp.kernels.ConstantKernel(46.81759185)*gp.kernels.Matern(length_scale=[0.47551771, 0.20404506])
    else:
        raise ValueError("Invalid kernel. Choices are RQ or MAT.")

    model = gp.GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=s**2)
    model.fit(X, y)

    # Get orthogonals of traj points
    range_deg = args.range / (np.radians(1.0) * 6369345)

    # Avoid orthogonals that are out of the map
    delta_idx = 2
    X = X[0:-delta_idx, :]

    orthogonal = get_ortogonal_points(X, 0, range_deg)

    orthogonals = np.zeros((X.shape[0], orthogonal.shape[0], orthogonal.shape[1]))

    for i in range(X.shape[0]):
        orthogonals[i] = get_ortogonal_points(X, i, range_deg)

    orthogonals = orthogonals[orthogonals[:,0,0] != 0]

    errors = np.zeros_like(orthogonals[:,:,1])

    test_gt_vals = RegularGridInterpolator((gt.lon, gt.lat), gt.chl[:,:,t_idx])

    for i in range(orthogonals.shape[0]):
        if np.count_nonzero(np.isnan(orthogonals[i])) > 0:
            continue
        y_pred = model.predict(orthogonals[i])
        gt_vals = test_gt_vals(orthogonals[i])
        errors[i] = (y_pred-gt_vals)*100/gt_vals

    # Plots
    n = np.linspace(0, orthogonals.shape[0]-1, orthogonals.shape[0])
    d = np.linspace(-args.range, args.range, orthogonals.shape[1])

    xx, yy = np.meshgrid(d, n, indexing='ij')
    plt.figure()
    p_rel_error = plt.pcolormesh(xx, yy, errors.T, vmin=-1, vmax=2, shading='auto', cmap='viridis')
    cbar_error = plt.colorbar(p_rel_error)
    cbar_error.set_label("Relative error [%]")
    plt.ylabel("Orthogonal index")
    plt.xlabel("Distance from trajectory [m]")
    plt.title("Stacked orthogonals relative error.")

    plt.figure()
    p_abs_rel_error = plt.pcolormesh(xx, yy, np.abs(errors.T), vmin=0, vmax=10, shading='auto', cmap='viridis')
    cbar_abserror = plt.colorbar(p_abs_rel_error)
    cbar_abserror.set_label("Relative error [%]")
    plt.ylabel("Orthogonal index")
    plt.xlabel("Distance from trajectory [m]")
    print("Stacked orthogonals absolute relative error  = %.3f %%." % (np.mean(np.abs(errors))))

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    gt_grid = np.meshgrid(gt.lon, gt.lat, indexing='ij')
    p_gt = plt.pcolormesh(gt_grid[0], gt_grid[1], gt.chl[:,:,t_idx], cmap='viridis', shading='auto', vmin=0, vmax=10)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cp = fig.colorbar(p_gt, cax=cax)
    cp.set_label("Chl [mm/mm3]")
    ax.plot(X[:, 0], X[:, 1], 'k-', linewidth=1)

    for ort in orthogonals:
        ax.plot(ort[:,0], ort[:,1], 'm-', linewidth=2.2)

    ax.set_xlabel("Longitude (ºE)")
    ax.set_ylabel("Latitude (ºN)")

    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)