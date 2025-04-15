"""
evals_dvals.py
====================================
Some functions to collect evals and dvals from multiple system parts at a given MPI rank

So far: Initital "collect_evals" and "collect_dvals"
for nonorthogonal TB
"""

import numpy as np
from sedacs.message import *

__all__ = ["collect_evals", "collect_dvals"]


def collect_evals(evalsOnRank, evals, verb=False):
    """
    Collects eigenvalues (evals) from multiple system parts at a given MPI rank.

    Paremeters
    ----------
    evalsOnRank : 1D numpy array
        Eigenvalues which have been collected at the current rank.
    evals : numpy 1D array
        Eigenvalues to be collected from each core+halo part.
    verb : bool, optional
        If set to True information is printed.

    Returns
    -------
    evalsOnRank : 1D numpy array
        Eigenvalues which have been collected at the current rank.
    """
    if verb:
        status_at("collect_evals", "Collecting evals")

    if evalsOnRank is None:
        evalsOnRank = evals
    else:
        evalsOnRank = np.append(evalsOnRank, evals)

    return evalsOnRank


def collect_dvals(dvalsOnRank, dvals, verb=False):
    """
    Collects dvals from multiple system parts at a given MPI rank.
    Notes: dvals are the contributions to the eigenvectors of the full system from each core+halo part.

    Paremeters
    ----------
    dvalsOnRank : 1D numpy array
        Dvals which have been collected at the current rank.
    dvals : numpy 1D array
        Dvals to be collected from each core+halo part.
    verb : bool, optional
        If set to True information is printed.

    Returns
    -------
    dvalsOnRank : 1D numpy array
        Dvals which have been at the current rank.
    """
    if verb:
        status_at("collect_dvals", "Collecting dvals")

    if dvalsOnRank is None:
        dvalsOnRank = dvals
    else:
        dvalsOnRank = np.append(dvalsOnRank, dvals)

    return dvalsOnRank
