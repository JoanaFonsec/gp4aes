import numpy as np
import netCDF4 as nc
import json
import h5py as h5
import subprocess
import os
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator


class ConfigData:
    def __init__(self, ll, ur, dx, start_date, end_date, dset_src, dt=None):
        self.ll = ll
        self.ur = ur
        self.dx = dx
        self.start_date = start_date
        self.end_date = end_date

        if not (dset_src == 'forecast' or dset_src == 'ocean_color'):
            raise ValueError("Unknown dataset source")

        self.dset_src = dset_src
        self.dt = dt


class AllData:
    def __init__(self, chl, lon, lat, time=None):
        self.chl = chl
        self.lon = lon
        self.lat = lat
        self.time = time


def parse_config_file(path):
    with open(path, "r") as f:
        config = json.load(f)

    if "dt" in config:
        return ConfigData(config["ll"], config["ur"], config["dx"],
                        config["start_date"], config["end_date"], config["dataset_source"], config["dt"])
    else:
        return ConfigData(config["ll"], config["ur"], config["dx"],
                        config["start_date"], config["end_date"], config["dataset_source"])


def load_data(path, data):
    print("Loading CMEMS forecast data...")

    root = nc.Dataset(path, 'r')

    time = root.variables["time"][()]

    if data.dset_src == 'forecast':
        # Forecast model time units are days since 1900-01-01
        time = time*86400 - 2208992400
    elif data.dset_src == 'ocean_color':
        # Forecast model time units are seconds since 1981-01-01
        time = time + 347155200

    if not (datetime.utcfromtimestamp(time[0]) <= datetime.utcfromtimestamp(data.start_date) <= \
            datetime.utcfromtimestamp(data.end_date) <= datetime.utcfromtimestamp(time[-1])):
        print("Time limits = [", datetime.utcfromtimestamp(time[0])," |", datetime.utcfromtimestamp(time[-1]), " ]")
        print("Time input = [", datetime.utcfromtimestamp(data.start_date)," |", datetime.utcfromtimestamp(data.end_date), " ]")
        raise ValueError("Inserted date is out of range")

    start_date_idx = int(np.argmin(np.abs(data.start_date - time)))
    end_date_idx = int(np.argmin(np.abs(data.end_date - time)))

    print("Operational time window: %s -- %s" %
            (datetime.utcfromtimestamp(time[start_date_idx]).ctime(),
             datetime.utcfromtimestamp(time[end_date_idx]).ctime()))

    time = time[start_date_idx:end_date_idx+1]

    lon = root.variables["lon"][()]
    lat = root.variables["lat"][()]

    if not (lat[0] <= data.ll[0] <= lat[-1] and \
            lat[0] <= data.ur[0] <= lat[-1] and \
            lon[0] <= data.ll[1] <= lon[-1] and \
            lon[0] <= data.ur[1] <= lon[-1]):
        raise ValueError("Inserted coordinates are out of range")

    # Interpolation margin
    delta_lat = np.diff(np.sort(lat)).max()
    delta_lon = np.diff(np.sort(lon)).max()

    lat_mask = np.where((data.ll[0] - delta_lat <= lat) & (lat <= data.ur[0] + delta_lat))[0]
    lon_mask = np.where((data.ll[1] - delta_lon <= lon) & (lon <= data.ur[1] + delta_lon))[0]

    lat = lat[lat_mask]
    lon = lon[lon_mask]

    # CHL concentration at surface
    if data.dset_src == 'forecast':
        chl = root.variables["chl"][start_date_idx:end_date_idx+1, 0, lat_mask[0]:lat_mask[-1]+1, lon_mask[0]:lon_mask[-1]+1]
    elif data.dset_src == 'ocean_color':
        chl = root.variables["CHL"][start_date_idx:end_date_idx+1, lat_mask[0]:lat_mask[-1]+1, lon_mask[0]:lon_mask[-1]+1]

    chl = np.where(chl.data == chl.fill_value, np.nan, chl.data)
    # chl = np.where(chl.data == chl.fill_value, 0, chl.data)

    print("Operational area: %.6fº -- %.6fº; %.6fº -- %.6fº" %
            (lon[0], lon[-1], lat[0], lat[-1]))

    return AllData(chl.T, lon, lat, time)


def process_data(data, dx, earth_radius):
    print("Processing data...")

    ddeg = dx / (np.radians(1.0) * earth_radius)

    lon_ax = np.arange(data.lon[0], data.lon[-1], step=ddeg)
    lat_ax = np.arange(data.lat[0], data.lat[-1], step=ddeg)

    t_ax = data.time

    xx, yy, tt = np.meshgrid(lon_ax, lat_ax, t_ax, indexing='ij')
    space = np.vstack([xx.ravel(), yy.ravel(), tt.ravel()]).T

    chl = RegularGridInterpolator((data.lon, data.lat, data.time), data.chl)(space)
    chl = np.reshape(chl, xx.shape)

    return AllData(chl, lon_ax, lat_ax, t_ax)

def write_data(path, data):
    print("Writing data to %s..." % (path))

    with h5.File(path, 'w') as f:
        ds = f.create_dataset("chl", data=data.chl)

        if data.time is not None:
            ds = f.create_dataset("time", data=data.time)
            ds.attrs.create("units", np.string_("unix_sec"))

        ds = f.create_dataset("lat", data=data.lat)
        ds.attrs.create("units", np.string_("degrees_north"))

        ds = f.create_dataset("lon", data=data.lon)
        ds.attrs.create("units", np.string_("degrees_east"))

def write_results(out_path,traj,measurement,gradient,chl,lon,lat,time,t_idx,delta_ref,time_step,meas_per, alpha_seek):
    print("Writing results to %s..." % (out_path))

    with h5.File(out_path, 'w') as f:
        f.create_dataset("traj", data=traj)
        f.create_dataset("chl", data=chl)
        f.create_dataset("lon", data=lon)
        f.create_dataset("lat", data=lat)
        f.create_dataset("time", data=time)
        f.create_dataset("measurement", data=measurement)
        f.create_dataset("gradient", data=gradient)
        f.attrs.create("t_idx", data=t_idx)
        f.attrs.create("delta_ref", data=delta_ref)
        f.attrs.create("time_step", data=time_step)
        f.attrs.create("meas_per", data=meas_per)
        f.attrs.create("alpha_seek", data=alpha_seek)

def add_land_mask(path):
    print("Adding land_mask...")

    mask_file = "grdlandmask.nc"

    with h5.File(path, 'r') as f:
        lon = f["lon"][()]
        lat = f["lat"][()]
        nt = f["time"].size

    nx = lon.size
    ny = lat.size

    gmt_cmdline = ["gmt", "grdlandmask",
                  "-G" + mask_file,
                  "-I" + str(nx) + "+n/" + str(ny) + "+n",
                  "-R" + str(lon[0]) + "/" + str(lon[-1]) +
                  "/" + str(lat[0]) + "/" + str(lat[-1]),
                  "-Df"]

    # Call Generic Mapping Tools
    subprocess.run(gmt_cmdline, check=True)

    land_mask = np.transpose(nc.Dataset(mask_file, "r")["z"][()])

    land_mask = np.moveaxis(np.tile(land_mask, (nt, 1, 1)), 0, 2)

    os.remove(mask_file)

    with h5.File(path, 'a') as f:
        ds = f.create_dataset("land_mask", data=land_mask)
        ds.attrs.create("units", np.string_("boolean: 1 = land"))