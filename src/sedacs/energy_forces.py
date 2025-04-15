"""
energy_forces.py
====================================
Some functions to collect energy and forces from multiple system parts at a given MPI rank
"""

import numpy as np
from sedacs.message import *

__all__ = ["collect_energy", "collect_forces"]


def collect_energy(energyOnRank, energy, verb=False):
    """
    Collects the energy from multiple system parts at a given MPI rank.

    Parameters
    ----------
    energyOnRank : float
        Energy contribution which have been collected at the current rank.
    energy : float
        Energy contribution from each core region to be collected.
    verb : bool, optional
        If set to True information is printed.

    Returns
    -------
    energyOnRank : float
        Energy contribution which have been collected at the current rank.
    """
    if verb:
        status_at("collect_energy", "Collecting energy")

    if energyOnRank is None:
        energyOnRank = np.zeros((1))

    energyOnRank[0] += energy

    return energyOnRank


def collect_forces(forcesOnRank, forces, part, nats, verb=False):
    """
    Collects the forces from multiple system parts at a given MPI rank.

    Parameters
    ----------
    forcesOnRank : 2D numpy array
        Forces which have been at the current rank.
    forces : 2D numpy array
        Forces acting each core region to be collected.
    part: list
        List of atom indices in the part.
    nats : int
        Number of total atoms in the system.
    verb : bool, optional
        If set to True information is printed.

    Returns
    -------
    forcesOnRank : 2D numpy array
        Forces which have been collected at the current rank.
    """
    if verb:
        status_at("collect_forces", "Collecting forces")

    if forcesOnRank is None:
        forcesOnRank = np.zeros((nats, 3))

    for i in range(len(part)):
        forcesOnRank[part[i]] = forces[i]

    return forcesOnRank
