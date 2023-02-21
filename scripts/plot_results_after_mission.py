from argparse import ArgumentParser
import h5py as h5
import numpy as np
import geopy.distance
import matplotlib.pyplot as plt
import matplotlib as mpl

import gp4aes.plotter.mission_plotter as plot_mission

# Read runtime arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the HDF5 file containing the processed data.")
    parser.add_argument("--pdf", action='store_true', help="Save plots in pdf format")
    parser.add_argument("--prefix",  type=str, help="Name used as prefix when saving plots.")

    return parser.parse_args()

def main(args):

    # Set plot format
    extension = 'png'
    if args.pdf:
        extension = 'pdf'

    # Set prefix for plot names
    plot_name_prefix = ""
    if args.prefix:
        plot_name_prefix = args.prefix + "-"

    # Read h5 file
    with h5.File(args.path, 'r') as f:
        lon = f["lon"][()]
        lat = f["lat"][()]
        chl = f["chl"][()]
        #time = f["time"][()]
        position = f["traj"][()]
        measurements = f["measurement"][()]
        gradient = f["gradient"][()]
        chl_ref = f.attrs["delta_ref"]
        time_step = f.attrs["time_step"]
        meas_per = f.attrs["meas_per"]
        t_idx = f.attrs["t_idx"]
        alpha_seek = f.attrs["alpha_seek"]

    # Call plotter class
    plotter = plot_mission.Plotter(position, gradient, measurements, lon, lat, chl[:,:,t_idx], chl_ref, meas_per, time_step, alpha_seek)

    ############################################ PRINTS
    # Attributes and length os variables
    print("delta_ref :", chl_ref)
    print("time_step :", time_step)
    print("meas_per :", meas_per)    
    print("alpha_seek :", alpha_seek)
    print('len(position) ', len(position[:, 0])-1, ' len(grad) ', len(gradient[:, 0]), ' len(measurements) ', len(measurements))
    
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