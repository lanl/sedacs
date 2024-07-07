"""Do MD timestep using GJF w half velocity"""

import numpy as np

__all__ = ["do_timestep"]


## Do time step integration.
# @brief
# @param
# @param
# @param
# @param
# @verb Verbosity
#
def do_timestep(coords, vels, forces, masses, gamma, dt, kT=None, verb=False):
    n = np.shape(coords[:, :])

    kT = 8.610e-5 * 300.0

    vels[:, :] = vels[:, :] + dt * forces[:, :] / 2.0 / masses[:, :]
    coords[:, :] = coords[:, :] + dt * vels[:, :] / 2.0

    if gamma > 0 and kT is not None:
        c = (1 - gamma * dt / 2.0) / (1 + gamma * dt / 2.0)
        vels[:, :] = c * vels[:, :] + np.sqrt((1 - c**2) * kT / masses[:, :]) * np.random.normal(0, 1, n)

    coords[:, :] = coords[:, :] + dt * vels[:, :] / 2.0
    vels[:, :] = vels[:, :] + dt * forces[:, :] / 2.0 / masses[:, :]

    return coords, vels
