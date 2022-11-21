
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import geopy.distance
from scipy.interpolate import RegularGridInterpolator

import gp4aes.util.parseh5 as parseh5
import gp4aes.estimator.GPR as gpr
import gp4aes.controller.front_tracking as controller
import gp4aes.util.parseh5 as h5
import gp4aes.plotter.plot_mission as plot_mission

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("lres_data_path", type=str, help="Path to the NETCDF4 file containing the lres dataset.")
    parser.add_argument("hres_data_path", type=str, help="Path to the NETCDF4 file containing the hres dataset.")
    parser.add_argument("lres_config_path", type=str, help="Path to the lres JSON config file.")
    parser.add_argument("hres_config_path", type=str, help="Path to the hres JSON config file.")
    parser.add_argument("out_path", type=str, help="Path to the HDF5 output file where processed data should be stored.")

    return parser.parse_args()

def main(args):

    ######################################### INIT
    ## VARIABLES
    earth_radius = 6369345
    timestamp = 1618610399
    clipped_area = np.array([61.412, 61.80, 20.771, 21.31])
    n_days = 3
    kernel_name = 'MAT'
    offset = 1

    ## READ DATASETS
    config_lres_data = parseh5.parse_config_file(args.lres_config_path)
    lres_data = parseh5.load_data(args.lres_data_path, config_lres_data)
    
    config_hres_data = parseh5.parse_config_file(args.hres_config_path)
    hres_data = parseh5.load_data(args.hres_data_path, config_hres_data)

    ## GET DATA
    chl = hres_data.chl
    time = hres_data.time
    lon = hres_data.lon
    lat = hres_data.lat

    ## TRAIN KERNEL 
    t_idx = np.argmin(np.abs(timestamp - lres_data.time))
    N = 1000 # train data
    N_meas = 500 # validation data
    n = 200 # test data
    s = 1e-3 # Approximately zero

    kernel_params = gpr.train_GP_model(lres_data.chl, lres_data.lat, lres_data.lon, s, N, N_meas, n, n_days, t_idx, offset, clipped_area, kernel_name)
   
    ######################################### RUN MISSION

    # Dynamics
    alpha_seek = 50
    alpha_follow = 1
    delta_ref = 7.45
    speed = 1.0 # 1m/s
    dynamics = controller.Dynamics(alpha_seek, alpha_follow, delta_ref, speed)
    range_m = 200
    std = 1e-3

    # Trajectory parameters
    init_towards = np.array([[21, 61.492]])
    init_coords = np.array([[20.925, 61.492]])
    time_step = 1
    meas_per = 1 # measurement period
    chl_ref = 7.45

    # Algorithm settings 
    n_iter = int(3e5) # 3e5
    n_meas = 200 # 125
    meas_filter_len = 3 # 3
    alpha = 0.97 # Gradient update factor, 0.95
    weights_meas = [0.2, 0.3, 0.5]
    init_flag = True

    ############ Initialize functions
    # Gaussian Process Regression
    est = gpr.GPEstimator(kernel_name, std, range_m, kernel_params)
    field = RegularGridInterpolator((lon, lat), chl[:,:,t_idx])

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

    for i in range(n_iter-1):
        if i % int(n_iter/100) == 0:
            print("Current iteration: %d" % i)
        if i % 100 == 0 and i != 0:
            print(" --- Error: {:.4f}".format(dynamics.delta_ref - measurements[-1]))

        ##### Take measurement
        
        val = field(position[-1,:]) + np.random.normal(0, est.s)
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
        control = dynamics(filtered_measurements[-1], filtered_gradient[-1,:], include_time=False)
        next_position = controller.next_position(position[-1, :],control)
        position = np.append(position, next_position, axis=0)

        if next_position[0, 1] > 61.64:
            break

    ############################################# END OF CYCLE ###################################

    h5.write_results(args.out_path,position,chl,lon,lat,time,measurements,filtered_gradient,t_idx,delta_ref,time_step,meas_per)

    # Set prefix for plot names
    extension = 'pdf'
    plot_name_prefix = ""

    # Call plotter class
    zoom=False
    plotter = plot_mission.Plotter(position, lon, lat, chl[:,:,t_idx], gradient, measurements, chl_ref, zoom, time, meas_per, time_step)

    # Average speed
    distances_between_samples = np.array([])
    for i in range(0,len(position[:, 0])-2):
        distance = geopy.distance.geodesic((position[i,1],position[i,0]), (position[i+1,1],position[i+1,0])).m
        distances_between_samples = np.append(distances_between_samples,distance)
    print("Average speed: {} m/s".format(np.mean(distances_between_samples)))

    ############################################ PLOTS
    # Mission overview
    fig_trajectory = plotter.mission_overview()
    fig_trajectory.savefig("plots/{}{}.{}".format(plot_name_prefix, "trajectory",extension),bbox_inches='tight')

    # Chl comparison
    fig_chl = plotter.chl_comparison()
    fig_chl.savefig("plots/{}{}.{}".format(plot_name_prefix, "measurements",extension),bbox_inches='tight')
    
    # Distance to front
    fig_distance = plotter.distance_to_front()
    fig_distance.savefig("plots/{}{}.{}".format(plot_name_prefix, "distance",extension),bbox_inches='tight')
    
    # Gradient comparison
    fig_gradient = plotter.gradient_comparison()
    fig_gradient.savefig("plots/{}{}.{}".format(plot_name_prefix, "gradient",extension),bbox_inches='tight')

    # Zoomed in overview
    fig_zoomed = plotter.zoomed_overview()
    fig_zoomed.savefig("plots/{}{}.{}".format(plot_name_prefix, "zoomed",extension),bbox_inches='tight')

    plt.show()



if __name__ == "__main__":
    args = parse_args()
    main(args)