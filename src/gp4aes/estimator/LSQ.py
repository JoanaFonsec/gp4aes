#!/usr/bin/env python3

import numpy as np

"""
2D Linear Squares estimation using numpy.linalg library

Parameters
----------
x: 2D coordinates array
y: Measurements on x coordinates
"""
def LSQ_estimation(x, y):
    if len(x[:,0]) != len(x[:,1]):
        raise ValueError("Estimation: Coordinates dimensions does not match.")

    if len(x[:,0]) != len(y):
        raise ValueError("Estimation: y must have the same length as x. Length of x:", len(x[:,0]), "Length of y:", len(y))

    A = np.hstack((x[:,[0]], x[:,[1]], np.ones((len(x[:,0]), 1))))

    y = y.reshape(-1,1)

    gradient_x, gradient_y, delta_zero = np.linalg.lstsq(A, y, rcond=None)[0].squeeze()

    return gradient_x, gradient_y