"""
charges.py
====================================
Some functions to compute charges

So far: Initital "get_charges" and "collect_charges"
for nonorthogonal TB
"""

import numpy as np
from sedacs.message import *

__all__ = ["get_charges", "collect_charges"]


def get_charges(rho, znuc, types, part, hindex, over=None, verb=False):
    """
    This function will return the point Mulliken populations
    for every atom in the system.

    Parameters
    ----------
    rho : numpy 2D array
        A 2D numpy array containing the elements of the Density matrix.
    znuc : int
        Number of valence electrons for every atom type.
    types : list of int
        Type index for every atom in the system.
    part : list
        Atom indices within the part (only cores).
    hindex : list
        Begining and ending of every index for atomic orbital block.
        Atom i will have its orbitals indexed in between (hindex[i],hindex[i+1])
    over : numpy 2D array
        A 2D numpy array containing the elements of the Overlap matrix.
    verb : bool
        If set to True information is printed.

    Returns
    -------
    2D numpy array
        A vector containing the Mulliquen charges is returned.

    Notes
    -----
    A vectorized version of this routine is needed.
    """
    ncores = len(part)
    if verb:
        status_at("get_charges", "Getting charges from density matrix")

    fullDiag = np.diag(rho)
    charges = np.zeros((ncores))

    nel = sum(znuc[types[:]])
    trRho = 0.0
    if over is None:
        for i in range(ncores):
            for ii in range(hindex[i], hindex[i + 1]):
                charges[i] = charges[i] + fullDiag[ii]
            charges[i] = 2.0 * charges[i] - znuc[types[i]]
    else:  # S x D
        aux = np.dot(over, rho)
        fullDiag = np.diag(aux)
        for i in range(ncores):
            charges[i] = 0.0
            for ii in range(hindex[i], hindex[i + 1]):
                charges[i] = charges[i] + fullDiag[ii]
                trRho = trRho + fullDiag[ii]
            charges[i] = 2.0 * charges[i] - znuc[types[i]]
    if verb:
        msg = "Total Charge for part= " + str(sum(charges))
        status_at("get_charges", msg)

    return charges


def collect_charges(chargesOnRank, charges, part, nats, verb=False):
    """
    Collects the charges from multiple system parts at a given MPI rank

    Given a vector representing the charges of a particular subsystem,
    denoted as charges, the values will be appropriately mapped to
    the charge vector that corresponds to the entire system, referred to as
    ``chargesOnRank``, by utilizing the information within the vector denoted as
    part.

    Parameters
    ----------
    chargesOnRank : 1D numpy array
        Full charge vetor on every rank.
    charges : 1D numpy array
        Charges to be collected.
    part : list
        List of atom indices in the part.
    nats : int
        Number of total atoms in the system.

    Returns
    -------
    1D numpy array
        Returns the same vector chargesOnRank with the
        added charges of the part

    """
    if verb:
        status_at("collect_charges", "Collecting charges")

    if chargesOnRank is None:
        chargesOnRank = np.zeros((nats))

    for i in range(len(part)):
        chargesOnRank[part[i]] = charges[i]

    return chargesOnRank
