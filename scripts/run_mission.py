try:
    import gp4aes.util.parseh5 as h5
except:
    import sys
    import os
    add_d = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(add_d+'/../src')
    import gp4aes.util.parseh5 as h5
import gp4aes.estimator.GPR as gpr
import gp4aes.controller.front_tracking as controller
from gp4aes.util.GeoGrid import read_h5_data

import numpy as np
import time
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the HDF5 file containing the processed data.")
    parser.add_argument("out_path", type=str, help="Path to the output HDF5 file containing trajectory data.")
    parser.add_argument("timestamp", type=int, help="Data date [UNIX].")
    parser.add_argument("kernel", type=str, help="RQ for Rational Quadratic kernel or MAT for Matérn one.")
    parser.add_argument("--include_time", action='store_true', help="Time-variant chl field.")
    parser.add_argument("--std", type=float, default=1e-3, help="Noise standard deviation.")
    parser.add_argument("--range", type=int, default=200, help="Estimation circle radius [m].")
    parser.add_argument("--kernel_params", nargs=3, type=float, help="3 kernel parameters.")

    return parser.parse_args()


def main(args):

    ############ Tunable parameters
    # Dynamics
    alpha_seek = 50
    alpha_follow = 1
    delta_ref = 7.45
    speed = 1.0 # 1m/s
    dynamics = controller.Dynamics(alpha_seek, alpha_follow, delta_ref, speed)

    # Trajectory parameters
    init_towards = np.array([[21, 61.492]])
    init_coords = np.array([[20.925, 61.492]])
    time_step = 1
    meas_per = 1 # measurement period

    # Algorithm settings (commented values are of trajectory for IROS paper)
    n_iter = int(3e5) # 3e5
    n_meas = 200 # 125
    meas_filter_len = 3 # 3
    alpha = 0.97 # Gradient update factor, 0.95
    weights_meas = [0.2, 0.3, 0.5]
    init_flag = False

    ############ Initialize functions
    # WGS84 grid
    grid = read_h5_data(args.path, args.timestamp, include_time=args.include_time)

    # Gaussian Process Regression
    est = gpr.GPEstimator(kernel=args.kernel, s=args.std, range_m=args.range, params=args.kernel_params)

    if args.include_time is not False:
        init_coords = np.array([[init_coords[0], init_coords[1], grid.time[grid.t_idx]]])
        init_heading = np.array([[init_towards[0, 0] - init_coords[0, 0], init_towards[0, 1] - init_coords[0, 1], 1]])

    else:
        init_heading = np.array([[init_towards[0, 0] - init_coords[0, 0], init_towards[0, 1] - init_coords[0, 1]]])
    
    # Main variables: position, measurements, gradient, filtered_measurements, filtered_gradient
    position = np.empty((0, init_coords.shape[1]))
    measurements = np.empty((0))
    gradient = np.empty((0,init_coords.shape[1]))
    filtered_measurements = np.empty((0, init_coords.shape[1]))
    filtered_gradient = np.empty((0, init_coords.shape[1]))

    # First value of the variables:
    position = np.append(position, init_coords, axis=0)

    ####################################### CYCLE ######################################################
    start = time.time()

    for i in range(n_iter-1):
        if i % int(n_iter/100) == 0:
            print("Current iteration: %d" % i)
        if i % 100 == 0 and i != 0:
            print(" --- Error: {:.4f}".format(dynamics.delta_ref - measurements[-1]))

        ##### Take measurement
        val = grid.field(position[-1,:]) + np.random.normal(0, est.s)
        if np.isnan(val):
            print("Warning: NaN value measured.")
            measurements = np.append(measurements, measurements[-1]) # Avoid plots problems
            break
        else:
            measurements = np.append(measurements, val)

        ##### Init state - From beginning until 5% tolerance from front
        if (i < n_meas or measurements[-1] < 0.95*dynamics.delta_ref) and init_flag is True:
            gradient = np.append(gradient, init_heading[[0], :2] / np.linalg.norm(init_heading[0, :2]), axis=0)
            filtered_gradient = np.append(filtered_gradient, gradient[[-1],:], axis=0)
            filtered_measurements = np.append(filtered_measurements,measurements[-1])

        ##### Estimation state - From reaching the front till the end of the mission
        else:
            if init_flag is True:
                print("Following the front...")
                init_flag = False

            filtered_measurements = np.append(filtered_measurements, 
                                        np.average(measurements[- meas_filter_len:], weights=weights_meas))

            # Estimate and filter gradient
            gradient_calculation = np.array(est.est_grad(position[-n_meas:],filtered_measurements[-n_meas:])).squeeze().reshape(-1, 2)
            gradient = np.append(gradient, gradient_calculation / np.linalg.norm(gradient_calculation), axis=0)
            filtered_gradient = np.append(filtered_gradient, filtered_gradient[[-2], :]*alpha + gradient[[-1], :]*(1-alpha), axis=0)

        ##### Calculate next position
        control = dynamics(filtered_measurements[-1], filtered_gradient[-1,:], include_time=args.include_time)
        next_position = controller.next_position(position[-1, :],control)
        position = np.append(position, next_position, axis=0)

        if not grid.is_within_limits(position[-1, :], include_time=args.include_time):
            print("Warning: trajectory got out of boundary limits.")
            break

        if next_position[0, 1] > 61.64:
            break

    ############################################# END OF CYCLE ###################################
    end = time.time()

    print("Time taken for the estimation:", end-start)

    h5.write_results(args.out_path,position,grid.data,grid.lon,grid.lat,grid.time,measurements,filtered_gradient,grid.t_idx,delta_ref,time_step,meas_per)

if __name__ == "__main__":
    args = parse_args()
    main(args)