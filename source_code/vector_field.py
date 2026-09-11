import random
import numpy as np
from opensimplex import OpenSimplex


class VecField:

  def __init__(self, config):

    self.dt = config["dt"]
    self.time = 0.0

    # initial wind vector [w_x, w_y]
    self.base_wind = np.array(config["initial_wind"], dtype=float)
    self.base_speed = float(np.linalg.norm(self.base_wind))
    self.base_angle = float(np.arctan2(self.base_wind[1], self.base_wind[0]))

    # hyperparameters for noise generation
    self.space_scale = config["space_scale"]
    self.time_scale = config["time_scale"]

    # % of base speed and max radians offset
    self.speed_variance = config["speed_variance"]
    self.max_angle_shift = np.radians(config["max_angle_shift"])

    # independent noise generators
    seed = random.randint(0, 2**31 - 1)
    self.noise_speed = OpenSimplex(seed=seed)
    self.noise_angle = OpenSimplex(seed=seed + 1000)

  def update(self):
    self.time += self.dt

  def get_vec(self, point2d):
    # scale env coordinates to noise space
    nx_coord = point2d[0] * self.space_scale
    ny_coord = point2d[1] * self.space_scale
    t_coord = self.time * self.time_scale

    # sample noise in range [-1, 1]
    n_speed = self.noise_speed.noise3(nx_coord, ny_coord, t_coord)
    n_angle = self.noise_angle.noise3(nx_coord, ny_coord, t_coord)

    # compute magnitude and angle variations
    speed = self.base_speed * (1.0 + n_speed * self.speed_variance)
    angle = self.base_angle + (n_angle * self.max_angle_shift)

    # reconstruct wind vector
    wind_vec = np.array([speed * np.cos(angle), speed * np.sin(angle)])

    return wind_vec