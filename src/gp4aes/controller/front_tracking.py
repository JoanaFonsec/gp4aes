import numpy as np
import math 


class Dynamics:
    def __init__(self, alpha_seek, alpha_follow, delta_ref, speed):
        self.alpha_seek = alpha_seek
        self.alpha_follow = alpha_follow
        self.delta_ref = delta_ref
        self.speed = speed

    def __call__(self, delta, grad, include_time=False):
        self.u_x = self.alpha_seek*(self.delta_ref - delta)*grad[0] \
                            - self.alpha_follow*grad[1]
        self.u_y = self.alpha_seek*(self.delta_ref - delta)*grad[1] \
                            + self.alpha_follow*grad[0]

        u = np.array([self.u_x, self.u_y])
        if include_time is False:
            return u * self.speed / np.linalg.norm(u)
        else:
            # Normalization still to be tested in TV conditions
            u_norm = u * self.speed / np.linalg.norm(u)
            return np.array([u_norm[0], u_norm[1], 1])


def next_position(position,control):
    # Calculate the increment in position, considering the earth radius
    earth_radius = 6369345
    next_position_x = position[0] + (control[0] / earth_radius) * (180 / math.pi) / math.cos(position[1] * math.pi/180)
    next_position_y = position[1] + (control[1] / earth_radius) * (180 / math.pi)

    return np.array([[next_position_x, next_position_y]])