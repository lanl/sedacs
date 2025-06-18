"""eVals dVals
Routines to build eVals and dVals. Typically
this will be done interfacing with an engine.

"""

import sys

from sedacs.types import ArrayLike
from sedacs.interface.pyseqm import pyseqmObjects
from sedacs.engine import Engine
import numpy as np

__all__ = ["get_eVals"]


def get_molSysData(eng: Engine,
                   sdc,
                   coords: ArrayLike,
                   symbols: ArrayLike,
                   atomTypes: ArrayLike,
                   do_large_tensors: bool = True,
                   device: str = 'cpu'):
    """
    Get the engin-specific molecular data.

    Parameters
    ----------
    eng : Engine
        The sedacs Engine object.
    sdc :
        The sedacs driver
    coords : ArrayLike (Natoms, 3)
        The Cartesian coordinates of the atoms in the system.
    symbols: ArrayLike
        The unique chemical elements in the structure.
    atomTypes: ArrayLike (Natoms, )
        The element type of each atom in the system.
    do_large_tensors : bool
        Whether to use large tensors.
    device : str
        The device to use.

    Returns
    -------
    Engine-dependent.
    """
    if eng.interface == "None":
        raise NotImplementedError("Interface not implemented")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        raise NotImplementedError("Interface not implemented")

    elif eng.interface == "PySEQM":
        return pyseqmObjects(sdc, coords,symbols,atomTypes, do_large_tensors=do_large_tensors, device=device)

    else:
        raise NotImplementedError("Interface not implemented")

