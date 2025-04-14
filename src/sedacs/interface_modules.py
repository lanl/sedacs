import ctypes
import os

import numpy as np
from sedacs.message import * 
import sys 
from sedacs.engine import Engine
from sedacs.globals import *
from sedacs.types import ArrayLike

# import the shared library
try:
    fortlibFileName = os.environ["PROXYA_FORTRAN_PATH"] + "/proxya_fortran.so"
#    fortlib = True
except Exception as e:
    fortlib = False

try: 
    pylibFileName = os.environ["PROXYA_PYTHON_PATH"]
    pylib = True
except Exception as e:
    pylib = False

# # try:
# #     #fortlib = ctypes.CDLL(fortlibFileName)
# #     get_hamiltonian_fortran = fortlib.proxya_get_hamiltonian
# #     get_density_matrix_fortran = fortlib.proxya_get_density_matrix
# # except Exception as e:
# #     fortlib = None
# # 
if( (not fortlib) and (not pylib)):
    print(fortlib,pylib)
    error_at("interface_modules","No specific fortran or python library exists")
    raise e

# try: 
    # from proxies.python.first_level import get_hamiltonian_proxy
    # from proxies.python.first_level import get_density_matrix_proxy
# except Exception as e:
    # pythlib = None
    # raise e

try: 
    from proxies.python.first_level import get_hamiltonian_proxy
    from proxies.python.first_level import get_density_matrix_proxy

    import inspect
    print(inspect.getfile(get_density_matrix_proxy))

    print("here")

    #from energy_and_forces import get_ppot_energy_expo_proxy
    # #from energy_and_forces import get_ppot_forces_expo_proxy
    # #from energy_and_forces import get_tb_forces_proxy
    from proxies.python.init_proxy import init_proxy_proxy
    from proxies.python.energy_and_forces import build_coul_ham_proxy
except Exception as e:
    pythlib = None
    raise e


__all__ = ["get_hamiltonian_module", "get_density_matrix_module",
        "get_ppot_energy_expo", "get_ppot_forces_expo", "init_proxy",
        "get_tb_forces_module","build_coul_ham_module"
        ]

#Initialize the proxy code
def init_proxy(symbols,orbs):
    """
    Initialize the proxy code.
    """

    init_proxy_proxy(symbols,orbs)
    

def build_coul_ham_module(eng,ham0,vcouls,types,charges,orbital_based,hindex,overlap=None,verb=False):
    if eng.name == "ProxyAPython":
        ham = build_coul_ham_proxy(ham0,vcouls,types,charges,orbital_based,hindex,overlap=overlap,verb=False)
    elif eng.name == "ProxyAFortran":
        error_at("build_coul_ham_module","ProxyAFortran version not implemented yet")
    elif eng.name == "ProxyAC":
        error_at("build_coul_ham_module","ProxyAC version not implemented yet")
    else:
        error_at("build_coul_ham_module","No specific engine type defined")
    
    return ham


def get_hamiltonian_module(eng: Engine,
                          coords: ArrayLike,
                          atomTypes: ArrayLike,
                          symbols: ArrayLike,
                          get_overlap: bool = True,
                          verb: bool = False):

    """
    Get the Hamiltonian of the system.

    Parameters
    ----------
    eng : Engine
        The engine to use.  
    coords : ArrayLike (Natoms, 3)  
        The coordinates of the atoms.
    atomTypes: ArrayLike (Natoms, )
        The element type of each atom in the system.
    symbols: ArrayLike
        The unique chemical elements in the structure.
    get_overlap : bool
        Whether to get the overlap matrix.
    verb : bool
        Whether to print verbose output.

    Returns
    -------
    hamiltonian : ArrayLike (Norb, Norb)
        The Hamiltonian matrix.

    overlap : ArrayLike (Norb, Norb)
        The overlap matrix. Only returned if get_overlap is True.
    """

    
    if eng.name == "ProxyAPython":
        if(get_overlap):
            hamiltonian, overlap = get_hamiltonian_proxy(coords,atomTypes,symbols,get_overlap=get_overlap,verb=verb)
        else:
            hamiltonian = get_hamiltonian_proxy(coords,atomTypes,symbols,get_overlap=get_overlap,verb=verb)

    elif eng.name == "ProxyAFortran":
        nats = len(coords[:, 0])
        norbs = nats

        coords_in = np.zeros(3 * nats)  # Vectorized coordinates
        for i in range(nats):
            coords_in[3 * i] = coords[i, 0]
            coords_in[3 * i + 1] = coords[i, 1]
            coords_in[3 * i + 2] = coords[i, 2]

        # Specify arguments type as a pointers
        get_hamiltonian_fortran.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_bool,
        ]
        # Passing a pointer to Fotran
        coords_ptr = coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        atomTypes_ptr = atomTypes.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        hamiltonian = np.zeros((norbs, norbs))
        overlap = np.zeros((norbs, norbs))
        ham_ptr = hamiltonian.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        over_ptr = overlap.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        err = get_hamiltonian_fortran(
            ctypes.c_int(nats), ctypes.c_int(norbs), coords_ptr, atomTypes_ptr, ham_ptr, over_ptr, ctypes.c_bool(verb)
        )
    else:
        error_at("get_hamiltonian_module","No specific engine type defined")

    if(get_overlap):
        return hamiltonian, overlap
    else:
        return hamiltonian



def get_density_matrix_modules(eng: Engine,
                               hamiltonian: ArrayLike,
                               nocc: int,
                               norbsInCore: Optional[int] = None,
                               method: str = "Diag",
                               accel: str = "No",
                               mu: Optional[float] = None,
                               etemp: float = 0.0,
                               overlap: Optional[ArrayLike] = None,
                               full_data: bool = False,
                               verb: bool = False) -> tuple[ArrayLike, ArrayLike, ArrayLike]:

    """
    Get the density matrix of the system.

    Parameters
    ----------
    eng : Engine
        The sedacs Engine object.
    hamiltonian : ArrayLike
        The Hamiltonian matrix.
    nocc : int
        The number of occupied orbitals.
    norbsInCore : Optional[int]
        The number of orbitals in the core.
    method : str
        The method to use.  
    accel : str
        The acceleration method to use.
    mu : Optional[float]
        The chemical potential.
    etemp : float
        The temperature.
    overlap : Optional[ArrayLike]   
        The overlap matrix.
    full_data : bool
        Whether to return the full data.
    verb : bool
        Whether to print verbose output.

    Returns 
    -------
    density_matrix : ArrayLike
        The density matrix.
    evals : ArrayLike
        The eigenvalues.
    dvals : ArrayLike
        The eigenvectors. ?


    """

    if eng.name == "ProxyAPython":
        method = eng.method
        accel = eng.accel
        if(full_data):
            density_matrix,evals,dvals = get_density_matrix_proxy(hamiltonian,nocc,norbsInCore=None,method=method,accel=accel,mu=mu, overlap=overlap,full_data=full_data,verb=False)
        else:
            density_matrix = get_density_matrix_proxy(hamiltonian,nocc,norbsInCore=None,method=method,accel=accel,mu=mu, overlap=overlap,full_data=full_data,verb=False)
    elif eng.name == "ProxyAFortran":
        # H needs to be flattened
        norbs = len(hamiltonian[:, 0])
        ht = hamiltonian.T
        # Specify arguments type as a pointers
        get_density_matrix_fortran.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_bool,
        ]
        # Passing a pointer to Fortran
        hamiltonian_ptr = hamiltonian.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        density_matrix = np.zeros((norbs, norbs))
        density_matrix_ptr = density_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        err = get_density_matrix_fortran(
            ctypes.c_int(norbs), ctypes.c_int(nocc), hamiltonian_ptr, density_matrix_ptr, ctypes.c_bool(verb)
        )
    else:
        error_at("get_density_matrix_module","No specific engine type defined")

    if(full_data):
        return density_matrix,evals,dvals
    else:
        return density_matrix


def get_ppot_energy_expo(coords: ArrayLike,
                         types: ArrayLike) -> float:
    """
    Get the potential energy from a potential.

    Parameters
    ----------
    coords : ArrayLike (Natoms, 3)
        The coordinates of the atoms.
    types: ArrayLike (Natoms, )
        The element type of each atom in the system.

    Returns
    -------
    energy : float
        The potential energy.
    """

    energy = get_ppot_energy_expo_proxy(coords,types)

    return energy


def get_ppot_forces_expo(coords: ArrayLike,
                         types: ArrayLike) -> ArrayLike:
    """
    Get the forces from a potential.

    Parameters
    ----------
    coords : ArrayLike (Natoms, 3)
        The coordinates of the atoms.
    types: ArrayLike (Natoms, )
        The element type of each atom in the system.

    Returns
    -------
    forces : ArrayLike (Natoms, 3)
        The forces on each atom.
    """

    forces = get_ppot_forces_expo_proxy(coords,types) 

    return forces


def get_tb_forces_module(ham: ArrayLike,
                         rho: ArrayLike,
                         charges: ArrayLike,
                         field: ArrayLike, # ?
                         coords: ArrayLike,
                         atomTypes: ArrayLike,
                         symbols: ArrayLike,
                         overlap: ArrayLike = None,
                         verb: bool = False):

    """

    Obtain forces from a tight binding model.

    Parameters
    ----------
    ham : ArrayLike (Norb, Norb)
        The Hamiltonian matrix.
    rho : ArrayLike (Norb, Norb)
        The density matrix.
    charges : ArrayLike (Natoms)
        The charges.
    field : ArrayLike
        The applied field.
    coords : ArrayLike (Natoms, 3)
        The coordinates of the atoms.
    symbols: ArrayLike
        The unique chemical elements in the structure.
    atomTypes: ArrayLike (Natoms, )
        The element type of each atom in the system.
    overlap : ArrayLike (Norb, Norb)
        The overlap matrix.
    verb : bool
        Whether to print verbose output.

    Returns
    -------
    forces : ArrayLike (Natoms, 3)
        The forces on each atom.
    """

    forces = get_tb_forces_proxy(ham,rho,charges,field,coords,atomTypes,symbols,overlap=None,verb=False)

    return forces

