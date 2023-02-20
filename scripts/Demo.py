import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import geopy.distance
from scipy.interpolate import RegularGridInterpolator

import gp4aes.util.parseh5 as parseh5
import gp4aes.estimator.GPR as gpr
from gp4aes.estimator.LSQ import LSQ_estimation
import gp4aes.controller.front_tracking as controller
import gp4aes.plotter.mission_plotter as plot_mission

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("lres_data_path", type=str, help="Path to the NETCDF4 file containing the lres dataset.")
    parser.add_argument("hres_data_path", type=str, help="Path to the NETCDF4 file containing the hres dataset.")
    parser.add_argument("lres_config_path", type=str, help="Path to the lres JSON config file.")
    parser.add_argument("hres_config_path", type=str, help="Path to the hres JSON config file.")
    parser.add_argument("out_path", type=str, help="Path to the HDF5 output file where processed data should be stored.")
    parser.add_argument("estimator", type=str, help="Estimator to be used: GP, LSQ.")

    return parser.parse_args()

def main(args):

    ######################################### INIT
    ## VARIABLES
    earth_radius = 6369345
    timestamp = 1618610399
    s = 1e-3 

    ## READ DATASETS
    config_lres_data = parseh5.parse_config_file(args.lres_config_path)
    data = parseh5.load_data(args.lres_data_path, config_lres_data)
    lres_data = parseh5.process_data(data, config_lres_data.dx, earth_radius)
    
    config_hres_data = parseh5.parse_config_file(args.hres_config_path)
    data = parseh5.load_data(args.hres_data_path, config_hres_data)
    hres_data = parseh5.process_data(data, config_hres_data.dx, earth_radius)

    t_idx = np.argmin(np.abs(timestamp - lres_data.time))

    ## SAVE DATA
    chl = hres_data.chl
    time = hres_data.time
    lon = hres_data.lon
    lat = hres_data.lat

    ## TRAIN KERNEL IF GP
    if args.estimator == 'GP':

        N = 1000 # train data
        N_meas = 500 # validation data
        n = 200 # test data
        n_days = 3
        offset = 1
        clipped_area = np.array([61.412, 61.80, 20.771, 21.31])
        kernel_name = 'MAT'

        # kernel_params = gpr.train_GP_model(lres_data.chl, lres_data.lat, lres_data.lon, s, N, N_meas, n, n_days, t_idx, offset, clipped_area, kernel_name)
        kernel_params = np.array([44.29588721, 0.54654887, 0.26656638])
        est = gpr.GPEstimator(kernel_name, s, kernel_params)

    ######################################### RUN MISSION

    ## DYNAMICS
    alpha_seek = 50
    alpha_follow = 1
    chl_ref = 7.45
    speed = 1.0 # 1m/s
    dynamics = controller.Dynamics(alpha_seek, alpha_follow, chl_ref, speed)
    t_idx = np.argmin(np.abs(timestamp - time))

    ## SETTINGS
    init_towards = np.array([[21, 61.492]])
    init_coords = np.array([[20.925, 61.492]])
    time_step = 1
    meas_per = 1 # measurement period
    
    n_iter = int(3e5) # 3e5
    n_meas = 20 # GP: 200, LSQ: 20
    meas_filter_len = 3 # 3
    alpha = 0.97 # Gradient update factor, 0.95
    weights_meas = [0.2, 0.3, 0.5]
    init_flag = True

    ## INIT FUNCTIONS
    field = RegularGridInterpolator((lon, lat), chl[:,:,t_idx])
    init_heading = np.array([[init_towards[0, 0] - init_coords[0, 0], init_towards[0, 1] - init_coords[0, 1]]])
    
    ## INIT VARIABLES
    position = np.empty((0, init_coords.shape[1]))
    measurements = np.empty((0))
    filtered_measurements = np.empty((0, init_coords.shape[1]))
    filtered_gradient = np.empty((0, init_coords.shape[1]))
    position = np.append(position, init_coords, axis=0)

    ####################################### CYCLE ######################################################

    for i in range(n_iter-1):
        if i % 4000 == 0:
            print("Current iteration: %d" % i)

        ##### Take measurement
        val = field(position[-1,:]) + np.random.normal(0, s)
        if np.isnan(val):
            print("Warning: NaN value measured.")
            measurements = np.append(measurements, measurements[-1]) # Avoid plots problems
            break
        else:
            measurements = np.append(measurements, val)

        ##### Init state - From beginning until 5% tolerance from front
        if (i < n_meas or measurements[-1] < 0.95*chl_ref) and init_flag is True:
            gradient = init_heading[[0], :2] / np.linalg.norm(init_heading[0, :2])
            filtered_gradient = np.append(filtered_gradient, gradient, axis=0)
            filtered_measurements = np.append(filtered_measurements,measurements[-1])

        ##### Estimation state - From reaching the front till the end of the mission
        else:
            if init_flag is True:
                print("Following the front...")
                init_flag = False

            filtered_measurements = np.append(filtered_measurements, np.average(measurements[- meas_filter_len:], weights=weights_meas))

            # Estimate and filter gradient
            if args.estimator == 'GP':
                gradient_calculation = np.array(est.est_grad(position[-n_meas:],filtered_measurements[-n_meas:])).squeeze().reshape(-1, 2)
            else:
                gradient_calculation = np.array(LSQ_estimation(position[-n_meas:],filtered_measurements[-n_meas:])).squeeze().reshape(-1, 2)
            gradient =  gradient_calculation / np.linalg.norm(gradient_calculation)
            filtered_gradient = np.append(filtered_gradient, filtered_gradient[[-1], :]*alpha + gradient*(1-alpha), axis=0)

        ##### Calculate next position
        control = dynamics(filtered_measurements[-1], filtered_gradient[-1,:], include_time=False)
        next_position = controller.next_position(position[-1, :],control)
        position = np.append(position, next_position, axis=0)

        if not (lon[0] <= position[-1, 0] <= lon[-1]) and not (lat[0] <= position[-1, 1] <= lat[-1]):
            print("Warning: trajectory got out of boundary limits.")
            break
        if next_position[0, 1] > 61.64: # 61.64
            break

    ############################################# END OF CYCLE ###################################

    parseh5.write_results(args.out_path,position,chl,lon,lat,time,measurements,filtered_gradient,alpha_seek,t_idx,chl_ref,time_step,meas_per, alpha_seek)

    # Call plotter class
    plotter = plot_mission.Plotter(position, lon, lat, chl[:,:,t_idx], filtered_gradient, filtered_measurements, chl_ref, meas_per, time_step, alpha_seek)
    extension = 'png'

    ############################################ PRINTS
    # Attributes and length os variables
    print("delta_ref :", chl_ref)
    print("time_step :", time_step)
    print("meas_per :", meas_per)    
    print("alpha_seek :", alpha_seek)
    print('len(position) ', len(position[:, 0])-1, ' len(grad) ', len(filtered_gradient[:, 0]), ' len(measurements) ', len(measurements))
    
    # Average speed
    distances_between_samples = np.array([])
    for i in range(0,len(position[:, 0])-2):
        distance = geopy.distance.geodesic((position[i,1],position[i,0]), (position[i+1,1],position[i+1,0])).m
        distances_between_samples = np.append(distances_between_samples,distance)
    print("Average speed: {} m/s".format(np.mean(distances_between_samples)))

     ############################################ PLOTS
    # Zoom 1 square
    lat_start1 = 61.525
    lat_end1 = 61.56
    lon_start1 = 21.1
    lon_end1 = 21.17
    # Zoom 2 square
    lon_start2 = 21.142
    lon_end2 = 21.152
    lat_start2 = 61.534  
    lat_end2 = 61.539
    # Zoom 3 square
    lon_start3 = 21.11
    lon_end3 = 21.12
    lat_start3 = 61.55
    lat_end3 = 61.555

    #a) Mission overview
    fig_trajectory = plotter.mission_overview(lon_start1,lon_end1,lat_start1,lat_end1)
    fig_trajectory.savefig("plots/{}.{}".format("big_map",extension),bbox_inches='tight')
    
    ######################## ZOOM 1
    #c) Gradient zoom1
    fig_gradient = plotter.gradient_comparison()
    fig_gradient.savefig("plots/{}.{}".format("gradient",extension),bbox_inches='tight')

    #d) Chl zoom1
    fig_chl = plotter.chl_comparison()
    fig_chl.savefig("plots/{}.{}".format("measurements",extension),bbox_inches='tight', dpi=300)

    # # b) Zoom1 map with gradient
    fig_zoom_gradient = plotter.zoom1(lon_start2,lon_end2,lat_start2,lat_end2,lon_start3,lon_end3,lat_start3,lat_end3)
    fig_zoom_gradient.savefig("plots/{}.{}".format("zoom1_map",extension),bbox_inches='tight')

    ####################### ZOOM 2
    #f) Control law zoom2
    fig_control = plotter.control_input(2)
    fig_control.savefig("plots/{}.{}".format("control2",extension),bbox_inches='tight')

    #e) Zoom 2 map with control law 
    fig_zoom_control = plotter.zoom2(lon_start2,lon_end2,lat_start2,lat_end2)
    fig_zoom_control.savefig("plots/{}.{}".format("zoom2_map",extension),bbox_inches='tight', dpi=300)

    # ######################## ZOOM 3
    #h) Control law zoom 3
    fig_control = plotter.control_input(3)
    fig_control.savefig("plots/{}.{}".format("control3",extension),bbox_inches='tight')

    #g) Zoom 3 map with control law 
    fig_zoom_control = plotter.zoom2(lon_start3,lon_end3,lat_start3,lat_end3)
    fig_zoom_control.savefig("plots/{}.{}".format("zoom3_map",extension),bbox_inches='tight', dpi=300)

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)