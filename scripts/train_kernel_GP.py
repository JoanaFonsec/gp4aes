#!/usr/bin/env python3

import numpy as np
import h5py as h5
from argparse import ArgumentParser

import gp4aes.estimator.GPR as gpr

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the HDF5 file containing the processed data.")
    parser.add_argument("timestamp", type=int, help="Time instant where to train/predict [UNIX].")
    parser.add_argument("kernel", type=str, help="RQ for Rational Quadratic kernel or MAT for Matérn one.")
    parser.add_argument("n_days", type=int, help="Number of days of training data to consider, previously to the test day.")
    parser.add_argument("--offset", type=int, default=1, help="Closests day offset.")
    parser.add_argument("--predict", action='store_true', help="Predict CHL density.")
    parser.add_argument("--clipped_area", type=float, nargs=4, help="If map contains NaN's, insert smaller operational area in the following order \
                                                                            LAT(min) LAT(max) LON(min) LON(max). Insert 1000 if some value is as default.")
    return parser.parse_args()


def main(args):
    with h5.File(args.path, "r") as f:
        chl = f["chl"][()]
        lat = f["lat"][()]
        lon = f["lon"][()]
        time = f["time"][()]

    t_idx = np.argmin(np.abs(args.timestamp - time))
    N = 1000 # train data
    N_meas = 500 # validation data
    n = 200 # test data
    s = 1e-3 # Approximately zero

    # Implement minimization of sum of negative log marginal likelihood of different datasets
    params = gpr.train_GP_model(chl, lat, lon, s, N, N_meas, n, args.n_days, t_idx, args.offset, args.clipped_area, kernel_name=args.kernel)
   
    # if args.predict:
    #     # Test predicitons
    #     xx, yy = np.meshgrid(x[0], x[1])
    #     x_stack = np.vstack([xx.ravel(), yy.ravel()]).T
    #     y_pred, std_pred = model.predict(x_stack, return_std=True)

    #     test_data_vals = RegularGridInterpolator((lon, lat), chl[:,:,t_idx])(x_stack)

    #     y_pred[np.isnan(test_data_vals)] = np.nan
    #     rel_error = np.abs(y_pred-test_data_vals)*100/test_data_vals

    #     # KPIs without NaNs
    #     y_pred_clean = y_pred[~np.isnan(test_data_vals)]
    #     test_data_vals_clean = test_data_vals[~np.isnan(test_data_vals)]
    #     av_rel_error = np.mean(np.abs(y_pred_clean-test_data_vals_clean)*100/test_data_vals_clean)
    #     av_std = np.mean(std_pred[~np.isnan(test_data_vals)])

    #     # Relative error
    #     print("Average relative error: %.3f%%" % np.mean(av_rel_error))
    #     print("Average std: %.3f" % av_std)

    #     # Write data on output file
    #     with h5.File(args.path, "w") as f:
    #         f.create_dataset("chl", data=chl)
    #         f.create_dataset("lat", data=lat)
    #         f.create_dataset("lon", data=lon)
    #         f.create_dataset("time", data=time)
    #         f.create_dataset("X", data=X)
    #         f.create_dataset("y", data=y)
    #         f.create_dataset("x", data=x)
    #         f.create_dataset("y_pred", data=y_pred)
    #         f.create_dataset("std_pred", data=std_pred)
    #         f.create_dataset("rel_error", data=rel_error)
    #         f.attrs.create("av_rel_error", data=av_rel_error)
    #         f.attrs.create("t_idx", data=t_idx)


if __name__ == "__main__":
    args = parse_args()
    main(args)
