
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import geopy.distance
from scipy.interpolate import RegularGridInterpolator

import gp4aes.util.parseh5 as parseh5
import gp4aes.estimator.GPR as gpr
import gp4aes.controller.front_tracking as controller
import gp4aes.plotter.mission_plotter as plot_mission

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
    kernel_name = 'MAT'

    ## READ DATASETS
    config_lres_data = parseh5.parse_config_file(args.lres_config_path)
    data = parseh5.load_data(args.lres_data_path, config_lres_data)
    lres_data = parseh5.process_data(data, config_lres_data.dx, earth_radius)
    
    config_hres_data = parseh5.parse_config_file(args.hres_config_path)
    data = parseh5.load_data(args.hres_data_path, config_hres_data)
    hres_data = parseh5.process_data(data, config_hres_data.dx, earth_radius)

    ## SAVE DATA
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
    n_days = 3
    offset = 1
    clipped_area = np.array([61.412, 61.80, 20.771, 21.31])

    # kernel_params = gpr.train_GP_model(lres_data.chl, lres_data.lat, lres_data.lon, s, N, N_meas, n, n_days, t_idx, offset, clipped_area, kernel_name)
    kernel_params = np.array([44.29588721, 0.54654887, 0.26656638])

    ######################################### RUN MISSION

    ## DYNAMICS
    alpha_seek = 50
    alpha_follow = 1
    delta_ref = 7.45
    speed = 1.0 # 1m/s
    dynamics = controller.Dynamics(alpha_seek, alpha_follow, delta_ref, speed)
    range_m = 200
    std = 1e-3
    t_idx = np.argmin(np.abs(timestamp - time))

    ## SETTINGS
    init_towards = np.array([[21, 61.492]])
    init_coords = np.array([[20.925, 61.492]])
    time_step = 1
    meas_per = 1 # measurement period
    chl_ref = 7.45
    n_iter = int(3e5) # 3e5
    n_meas = 200 # 125
    meas_filter_len = 3 # 3
    alpha = 0.97 # Gradient update factor, 0.95
    weights_meas = [0.2, 0.3, 0.5]
    init_flag = True

    ## INIT FUNCTIONS
    est = gpr.GPEstimator(kernel_name, std, range_m, kernel_params)
    field = RegularGridInterpolator((lon, lat), chl[:,:,t_idx])
    init_heading = np.array([[init_towards[0, 0] - init_coords[0, 0], init_towards[0, 1] - init_coords[0, 1]]])
    
    ## INIT VARIABLES
    position = np.empty((0, init_coords.shape[1]))
    measurements = np.empty((0))
    gradient = np.empty((0,init_coords.shape[1]))
    filtered_measurements = np.empty((0, init_coords.shape[1]))
    filtered_gradient = np.empty((0, init_coords.shape[1]))
    control_law = np.empty((0, init_coords.shape[1]))

    position = np.append(position, init_coords, axis=0)

    ####################################### CYCLE ######################################################

    for i in range(n_iter-1):
        if i % int(n_iter/100) == 0:
            print("Current iteration: %d" % i)

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
        control_law = np.append(control_law, control.reshape((-1,2)), axis=0)
        next_position = controller.next_position(position[-1, :],control)
        position = np.append(position, next_position, axis=0)

        if not (lon[0] <= position[-1, 0] <= lon[-1]) and not (lat[0] <= position[-1, 1] <= lat[-1]):
            print("Warning: trajectory got out of boundary limits.")
            break
        if next_position[0, 1] > 61.64:
            break

    ############################################# END OF CYCLE ###################################

    parseh5.write_results(args.out_path,position,chl,lon,lat,time,measurements,filtered_gradient,control_law,t_idx,delta_ref,time_step,meas_per)

    ## INIT PLOTTER
    zoom = False
    plotter = plot_mission.Plotter(position, lon, lat, chl[:,:,t_idx], filtered_gradient, measurements, control_law, chl_ref, zoom, meas_per, time_step)
    extension = 'pdf'
    plot_name_prefix = ""

    ## AVG SPEED
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

    # Control law
    fig_control = plotter.control_input()
    fig_control.savefig("plots/{}{}.{}".format(plot_name_prefix, "control",extension),bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)