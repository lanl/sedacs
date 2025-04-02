from sedacs.interface.pyseqm import get_overlap_pyseqm
from sedacs.types import ArrayLike
from sedacs.engine import Engine

import torch

__all__ = ["get_overlap"]


def get_overlap(eng: Engine,
                coord,
                symbols,
                types,
                hindex) -> ArrayLike:
    """

    Builds and returns the overlap for a given chemical or crystalline system. This needs to be implemented for 
    each interfaced engine uniquely. This function can either call external routines, or manually
    compute overlaps from electronic information provided by the external code.

    Parameters
    ----------
    eng: sedacs.eng.Engine
        Sedacs Engine object which contains the relevant information for the chemical system and 
        calculation being carried out.
    coord: ArrayLike
        (Natoms, 3) numpy array or torch Tensor representing the Cartesian coordinates of the system.
    symbols: ArrayLike
        The unique chemical elements in the structure.
    types: ArrayLike (Natoms, )
        The element type of each atom in the system.
    hindex: ArrayLike
        List, torch Tensor, or numpy arrray containing the indices of Hydrogen atoms. This is for 
        interfaced codes (such as PYSEQM) which pad the p-orbitals of Hydrogens in some representations.
        TODO: Make this parameter optional, as it likely isn't ever needed outside the context of the
        PYSEQM interface.

    Returns
    -------
    S: ArrayLike
        The overlap matrix (consistent with the format required/specified by your engine's interface).

    """


    # Tight interface using modules or an external code compiled as a library
    if eng.interface == "None":
        raise NotImplementedError(f"{eng.interface} is not implemented, select from the implemented engine types")
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        raise NotImplementedError(f"{eng.interface} is not implemented, select from the implemented engine types")

    elif eng.interface == "PySEQM":

        with torch.no_grad():
            S = get_overlap_pyseqm(coords, symbols, types, hindex).detach().numpy()
            return S
        
    else:
        raise NotImplementedError(f"{eng.interface} is not implemented, select from the implemented engine types")

    # TODO, not sure what this is, but the code is unreachable.
    return rho
