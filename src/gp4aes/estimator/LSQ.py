#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import h5py as h5
from argparse import ArgumentParser

import time

class Estimator:
    def __init__(self, init_params):
        self.alpha = init_params[0]
        self.beta = init_params[1]
        self.delta_zero = init_params[2]

        self.f = lambda x: self.alpha*x[0] + self.beta*x[1] + self.delta_zero


    """
    2D Linear Squares estimation using numpy.linalg library

    Parameters
    ----------
    x: 2D coordinates array
    y: Measurements on x coordinates
    """
    def estimate(self, x, y):
        if len(x[0]) != len(x[1]):
            raise ValueError("Estimation: Coordinates dimensions does not match.")

        if len(x[0]) != len(y):
            raise ValueError("Estimation: y must have the same length as x.")

        A = np.vstack([x[0], x[1], np.ones(len(x[0]))]).T

        y = y[:, np.newaxis]

        self.alpha, self.beta, self.delta_zero = np.linalg.lstsq(A, y, rcond=None)[0].squeeze()

        return self.alpha, self.beta, self.delta_zero


def euler(x0, step, speed, dynamics, grid, init_heading):
    print("Calculating trajectory without the estimation algorithm...")
    n_iter = 400000

    x0 = [x0[0], x0[1]]

    traj = np.zeros((n_iter, len(x0)))
    delta = np.zeros(n_iter)
    out_grad = np.zeros((n_iter, 2))

    traj[0] = x0
    init_heading = np.array([init_heading[0] - x0[0], init_heading[1] - x0[1]])

    grad = np.gradient(grid.data[:,:,grid.t_idx])
    norm = np.sqrt(grad[0]**2 + grad[1]**2)
    gradient = (RegularGridInterpolator((grid.lon, grid.lat), grad[0]/norm),
                RegularGridInterpolator((grid.lon, grid.lat), grad[1]/norm))

    for i in range(n_iter-1):
        if i % 10000 == 0:
            print("Current iteration: %d" % i)

        x_curr = traj[i]
        delta[i] = grid.field(x_curr) + np.random.normal(0, 5e-3)

        grad = (gradient[0]((x_curr[0], x_curr[1])), gradient[1]((x_curr[0], x_curr[1])))
        out_grad[i+1] = grad

        control = dynamics(delta[i], grad, include_time=False)

        if np.sqrt(control[0]**2 + control[1]**2) >= speed:
            control[0] = speed * control[0] / np.sqrt(control[0]**2 + control[1]**2)
            control[1] = speed * control[1] / np.sqrt(control[0]**2 + control[1]**2)

        traj[i+1] = x_curr + step*control

        if not grid.is_within_limits(traj[i+1]):
            print("Warning: trajectory got out of boundary limits.")
            break


    return traj[:i], delta[:i], out_grad[:i]


def euler_est(x0, step, speed, dynamics, grid, estimator, init_heading, include_time=False):
    print("Calculating trajectory estimating the front...")

    n_iter = int(5e6)
    n_meas = 75
    meas_per = int(3 / step)
    estimation_trigger_val = (n_meas-1) * meas_per

    if include_time is not False:
        x0 = [x0[0], x0[1], grid.time[grid.t_idx]]
        init_heading = np.array([init_heading[0] - x0[0], init_heading[1] - x0[1], 1])

    else:
        init_heading = np.array([init_heading[0] - x0[0], init_heading[1] - x0[1]])

    traj = np.zeros((n_iter, len(x0)))
    grad = np.zeros((n_iter, 2))

    x_meas = np.zeros((int(np.ceil(n_iter / meas_per)), len(x0)))
    measurements = np.zeros(int(np.ceil(n_iter / meas_per)))

    traj[0] = x0

    # Init state control law
    control = init_heading

    if np.sqrt(control[0]**2 + control[1]**2) >= speed:
        control[0] = speed * control[0] / np.sqrt(control[0]**2 + control[1]**2)
        control[1] = speed * control[1] / np.sqrt(control[0]**2 + control[1]**2)

    # Init state
    meas_index = 0
    for i in range(estimation_trigger_val+1):
        if i % int(n_iter/100) == 0:
            print("Current iteration: %d" % i)

        offset = i % meas_per

        if offset == 0:
            x_meas[meas_index] = traj[i]
            measurements[meas_index] = grid.field(x_meas[meas_index]) + np.random.normal(0, 0.005)
            grad[i] = init_heading[:2]

            meas_index = meas_index + 1
        else:
            grad[i] = grad[i - offset]

        traj[i+1] = traj[i] + step*control

    # Estimation state
    for i in range(estimation_trigger_val+1, n_iter-1):
        if i % int(n_iter/100) == 0:
            print("Current iteration: %d" % i)

        offset = i % meas_per

        if offset == 0:
            x_meas[meas_index] = traj[i]
            measurements[meas_index] = grid.field(x_meas[meas_index]) + np.random.normal(0, 0.005)
            chl_field = estimator.estimate((x_meas[meas_index-n_meas:meas_index+1, 0], x_meas[meas_index-n_meas:meas_index+1, 1]),\
                                            measurements[meas_index-n_meas:meas_index+1])

            grad[i] = chl_field[:2] / np.sqrt(chl_field[0]**2 + chl_field[1]**2)
            control = dynamics(measurements[meas_index], grad[i], include_time=include_time)

            meas_index = meas_index + 1

            if np.sqrt(control[0]**2 + control[1]**2) >= speed:
                control[0] = speed * control[0] / np.sqrt(control[0]**2 + control[1]**2)
                control[1] = speed * control[1] / np.sqrt(control[0]**2 + control[1]**2)
        else:
            grad[i] = grad[i - offset]

        traj[i+1] = traj[i] + step*control

        if not grid.is_within_limits(traj[i+1, :], include_time):
            print("Warning: trajectory got out of boundary limits.")
            break

    return traj[:i], measurements[:meas_index], grad[:i]
