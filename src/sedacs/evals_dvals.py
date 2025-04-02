"""
evals_dvals.py
====================================
Some functions to compute evals and dvals

So far: Initital "collect_evals" and "collect_dvals"
for nonorthogonal TB
"""

import numpy as np
from sedacs.message import *

__all__ = ["collect_evals", "collect_dvals"]


def collect_evals(evalsOnRank, evals, verb=False):
    """
    The function will collect eigenvalues (evals) from each core+halo part at the current rank.

    Paremeters
    ----------
    evalsOnRank : numpy 1D array
        A 1D numpy array containing the eigenvalues at the current rank.
    evals : numpy 1D array
        A 1D numpy array containing the eigenvalues to be collected from each core+halo part.
    verb : bool, optional
        If set to True information is printed.

    Returns
    -------
    evalsOnRank : numpy 1D array
        A 1D numpy array containing the eigenvalues at the current rank.
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
    The function will collect the contributions (dvals) from each core+halo part to the eigenvectors of the full system at the current rank.

    Paremeters
    ----------
    dvalsOnRank : numpy 1D array
        A 1D numpy array containing the dvals at the current rank.
    dvals : numpy 1D array
        A 1D numpy array containing the dvals to be collected from each core+halo part.
    verb : bool, optional
        If set to True information is printed.

    Returns
    -------
    dvalsOnRank : numpy 1D array
        A 1D numpy array containing the dvals at the current rank.
    """
    if verb:
        status_at("collect_dvals", "Collecting dvals")

    if dvalsOnRank is None:
        dvalsOnRank = dvals
    else:
        dvalsOnRank = np.append(dvalsOnRank, dvals)

    return dvalsOnRank
