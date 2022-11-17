#!/usr/bin/env python3
import gp4aes.util.parseh5 as parseh5

from argparse import ArgumentParser
import matplotlib.pyplot as plt

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("data_path", type=str, help="Path to the NETCDF4 file containing the dataset.")
    parser.add_argument("config_path", type=str, help="Path to the JSON config file.")
    parser.add_argument("out_path", type=str, help="Path to the HDF5 output file where processed data should be stored.")
    parser.add_argument("--earth_radius", type=float, default=6369345, help="Earth average radius [m].")
    parser.add_argument("--raw", action="store_true", help="Write raw data in ouput file.")
    parser.add_argument("--add_land_mask", action="store_true", help="Add land mask from GSHHG dataset.")

    return parser.parse_args()

def main(args):
    config_data = parseh5.parse_config_file(args.config_path)

    data = parseh5.load_forecast_data(args.data_path, config_data)

    if args.raw:
        parseh5.write_data(args.out_path, data)
    if not args.raw:

        # if it is hourly then run this so that the time step is given in minutes:
        #dt = config_data.dt * 60

        data_proc = parseh5.process_forecast_data(data, config_data.dx, args.earth_radius, dt=dt)
        parseh5.write_data(args.out_path, data_proc)

    if args.add_land_mask:
        parseh5.add_land_mask(args.out_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)