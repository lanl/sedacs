import ctypes
import os

import numpy as np
from sedacs.message import *
import sys
from sedacs.engine import Engine
from sedacs.globals import *

# import the shared library
fortlibFileName = os.environ["PROXYA_FORTRAN_PATH"] + "/proxya_fortran.so"
pylibFileName = os.environ["PROXYA_PYTHON_PATH"]
sys.path.append(pylibFileName)

try:
    fortlib = ctypes.CDLL(fortlibFileName)
    get_hamiltonian_fortran = fortlib.proxya_get_hamiltonian
    get_density_matrix_fortran = fortlib.proxya_get_density_matrix
except Exception as e:
    fortlib = None
    raise e

try:
    from first_level import get_hamiltonian_proxy
    from first_level import get_density_matrix_proxy
    from first_level import get_ppot_energy_expo_proxy
    from first_level import get_ppot_forces_expo_proxy
    from first_level import get_tb_forces_proxy
    from first_level import init_proxy_proxy
except Exception as e:
    pythlib = None
    raise e


__all__ = ["get_hamiltonian_module", "get_density_matrix_module",
        "get_ppot_energy_expo", "get_ppot_forces_expo", "init_proxy",
        "get_tb_forces_module"
        ]

#Initialize the proxy code
def init_proxy(symbols,orbs):
    init_proxy_proxy(symbols,orbs)



#def get_hamiltonian_proxy(*args, **kwargs):
#    raise NotImplementedError("implement this in an external module!")


def get_hamiltonian_module(eng,coords,atomTypes,symbols,get_overlap=True,verb=False):


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


def get_density_matrix_modules(eng,hamiltonian,nocc,method="Diag",accel="No",mu=None,elect_temp=0.0,overlap=None, verb=False):
    if eng.name == "ProxyAPython":
        method = eng.method
        accel = eng.accel
        density_matrix = get_density_matrix_proxy(hamiltonian, nocc,method=method,accel=accel,mu=mu, overlap=overlap,  verb=False)
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

    return density_matrix


def get_ppot_energy_expo(coords,types):

    energy = get_ppot_energy_expo_proxy(coords,types)

    return energy


def get_ppot_forces_expo(coords,types):

    forces = get_ppot_forces_expo_proxy(coords,types)

    return forces


def get_tb_forces_module(coords,types):

    forces = get_ppot_forces_expo_proxy(coords,types)

    return forces
