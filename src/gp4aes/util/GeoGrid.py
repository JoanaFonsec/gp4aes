import numpy as np
import math 
import h5py as h5
from scipy.interpolate import RegularGridInterpolator


class GeoGrid:
    def __init__(self, data, lon, lat, time, t_idx, include_time=False):
        self.data = data
        self.lon = lon
        self.lat = lat
        self.time = time
        self.t_idx = t_idx

        if include_time is False:
            self.field = RegularGridInterpolator((self.lon, self.lat), self.data[:,:,t_idx])
        else:
            self.field = RegularGridInterpolator((self.lon, self.lat, self.time), self.data)

    def is_within_limits(self, x, include_time=False):
        if include_time is False:
            if (self.lon[0] <= x[0] <= self.lon[-1]) and (self.lat[0] <= x[1] <= self.lat[-1]):
                return True
        else:
            if (self.lon[0] <= x[0] <= self.lon[-1]) and (self.lat[0] <= x[1] <= self.lat[-1]) and (self.time[0] <= x[2] <= self.time[-1]):
                return True

def read_h5_data(path, timestamp, include_time=False):
    with h5.File(path, 'r') as f:
        chl = f["chl"][()]
        lat = f["lat"][()]
        lon = f["lon"][()]
        time = f["time"][()]

    t_idx = np.argmin(np.abs(timestamp - time))

    return GeoGrid(chl, lon, lat, time, t_idx, include_time=include_time)