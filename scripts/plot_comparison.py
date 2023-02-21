from argparse import ArgumentParser
import h5py as h5
import numpy as np
import geopy.distance
import matplotlib.pyplot as plt

import gp4aes.plotter.comparison_plotter as plot_comparison

# Read runtime arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path1", type=str, help="Path to the HDF5 file containing the processed data 1.")
    parser.add_argument("path2", type=str, help="Path to the HDF5 file containing the processed data 2.")
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

    # Read h5 file of data 1
    with h5.File(args.path1, 'r') as f:
        lon = f["lon"][()]
        lat = f["lat"][()]
        chl = f["chl"][()]
        #time = f["time"][()]
        position1 = f["traj"][()]
        measurements1 = f["measurement"][()]
        gradient1 = f["gradient"][()]
        chl_ref = f.attrs["delta_ref"]
        time_step = f.attrs["time_step"]
        meas_per = f.attrs["meas_per"]
        t_idx = f.attrs["t_idx"]

    # Read h5 file of data 2
    with h5.File(args.path2, 'r') as f:
        position2 = f["traj"][()]
        measurements2 = f["measurement"][()]
        gradient2 = f["gradient"][()]

    # Call plotter class
    plotter = plot_comparison.Plotter(position1, gradient1, measurements1, position2, gradient2, measurements2, lon, lat, chl[:,:,t_idx], chl_ref, meas_per, time_step)

    ############################################ PRINTS
    # Attributes and length os variables
    print("delta_ref :", chl_ref)
    print("time_step :", time_step)
    print("meas_per :", meas_per)    
    print('len(position) ', len(position1[:, 0]), ' len(grad) ', len(gradient1[:, 0]), ' len(measurements) ', len(measurements1))
    
    # Average speed
    distances_between_samples1 = np.array([])
    distances_between_samples2 = np.array([])
    for i in range(0,min(len(position1[:, 0]),len(position2[:, 0]))-2):
        distance = geopy.distance.geodesic((position1[i,1],position1[i,0]), (position1[i+1,1],position1[i+1,0])).m
        distances_between_samples1 = np.append(distances_between_samples1,distance)
        distance = geopy.distance.geodesic((position2[i,1],position2[i,0]), (position2[i+1,1],position2[i+1,0])).m
        distances_between_samples2 = np.append(distances_between_samples2,distance)
    print("Average speed for GP: {} m/s".format(np.mean(distances_between_samples1)), "Average speed for LSQ: {} m/s".format(np.mean(distances_between_samples2)))

    #Chl zoom1
    fig_chl = plotter.chl_comparison()
    fig_chl.savefig("plots/{}.{}".format("measurements",extension),bbox_inches='tight', dpi=300)

    #Gradient zoom1
    fig_gradient = plotter.gradient_comparison()
    fig_gradient.savefig("plots/{}.{}".format("gradient",extension),bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)