#!/usr/bin/env python3

from matplotlib import cm
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to HDF5 output file containig processed data")
    parser.add_argument("timestamp", type=int, help="Time instant to plot the mat [UNIX timestamp]")
    parser.add_argument("--c_lev", type=float, nargs='+', help="Contour levels to plot - include all values to plot a contour line at.")

    return parser.parse_args()


def main(args):
    with h5.File(args.path, "r") as f:
        lat = f["lat"][()]
        lon = f["lon"][()]
        chl = f["chl"][()]
        time = f["time"][()]

        if "land_mask" in f:
            land_mask = f["land_mask"][()]
            chl[land_mask > 0.5] = np.nan

    grid = np.meshgrid(lon, lat, indexing='ij')

    t_idx = np.argmin(np.abs(args.timestamp - time))

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    p = ax.pcolormesh(grid[0], grid[1], chl[:, :, t_idx], cmap='viridis', shading='auto', vmin=0, vmax=10)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cp = fig.colorbar(p, cax=cax)
    # cp = fig.colorbar(p)
    cp.set_label("Chl a density [mm/mm3]")

    if args.c_lev:
         ax.contour(grid[0], grid[1], chl[:,:,t_idx], levels=args.c_lev)

    ax.set_xlabel("Longitude (ºE)")
    ax.set_ylabel("Latitude (ºN)")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)