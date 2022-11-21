#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import gp4aes.util.parseh5 as parseh5

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("data_path", type=str, help="Path to the NETCDF4 file containing the dataset.")
    parser.add_argument("config_path", type=str, help="Path to the JSON config file.")
    parser.add_argument("out_path", type=str, help="Path to the HDF5 output file where processed data should be stored.")
    parser.add_argument("timestamp", type=int, help="Time instant to plot the mat [UNIX timestamp]")
    parser.add_argument("--earth_radius", type=float, default=6369345, help="Earth average radius [m].")
    parser.add_argument("--raw", action="store_true", help="Write raw data in ouput file.")
    parser.add_argument("--add_land_mask", action="store_true", help="Add land mask from GSHHG dataset.")

    return parser.parse_args()

def main(args):
    ## READ DATASET
    config_data = parseh5.parse_config_file(args.config_path)

    data = parseh5.load_data(args.data_path, config_data)

    if args.raw:
        parseh5.write_data(args.out_path, data)
    if not args.raw:
        data_proc = parseh5.process_data(data, config_data.dx, args.earth_radius)
        parseh5.write_data(args.out_path, data_proc)

    if args.add_land_mask:
        parseh5.add_land_mask(args.out_path)

    ### PLOT IT
    if args.add_land_mask:
        data.chl[args.add_land_mask > 0.5] = np.nan

    grid = np.meshgrid(data.lon, data.lat, indexing='ij')

    t_idx = np.argmin(np.abs(args.timestamp - data.time))

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    p = ax.pcolormesh(grid[0], grid[1], data.chl[:, :, t_idx], cmap='viridis', shading='auto', vmin=0, vmax=10)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cp = fig.colorbar(p, cax=cax)
    cp.set_label("Chl a density [mm/mm3]")
    ax.contour(grid[0], grid[1], data.chl[:,:,t_idx], levels=7)

    ax.set_xlabel("Longitude (ºE)")
    ax.set_ylabel("Latitude (ºN)")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)