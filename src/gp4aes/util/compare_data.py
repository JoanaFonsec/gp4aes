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
    parser.add_argument("path_gt", type=str, help="Path to the HDF5 file containing the processed GT data.")
    parser.add_argument("path_test", type=str, help="Path to the HDF5 file containing the processed data to be compared with GT.")

    return parser.parse_args()


def main(args):
    with h5.File(args.path_gt, "r") as f:
        gt = GridData(f["chl"][()], f["lat"][()], f["lon"][()], f["time"][()])

    with h5.File(args.path_test, "r") as f:
        x = f["x"][()]
        y_pred = f["y_pred"][()]
        t_idx = f.attrs["t_idx"]

    xx, yy = np.meshgrid(x[0], x[1])
    x_stack = np.vstack([xx.ravel(), yy.ravel()]).T

    data_x_mask = np.where((gt.lon[0] < x_stack[:,0]) & (x_stack[:,0] < gt.lon[-1]) \
                                & (gt.lat[0] < x_stack[:,1]) & (x_stack[:,1] < gt.lat[-1]))[0]
    x_stack = x_stack[data_x_mask]
    y_pred = y_pred[data_x_mask]

    test_data_vals = RegularGridInterpolator((gt.lon, gt.lat), gt.chl[:,:,t_idx])(x_stack)
    rel_error = np.abs(y_pred[~np.isnan(test_data_vals)]-test_data_vals[~np.isnan(test_data_vals)])*100/test_data_vals[~np.isnan(test_data_vals)]
    av_rel_error = np.mean(rel_error[~np.isnan(rel_error)])
    print("Average relative error: %.3f%%" % av_rel_error)

if __name__ == "__main__":
    args = parse_args()
    main(args)