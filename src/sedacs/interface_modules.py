import ctypes
import os

import numpy as np

# import the shared library
fortlibFileName = os.environ["PROXYA_FORTRAN_PATH"] + "/proxya_fortran.so"

try:
    fortlib = ctypes.CDLL(fortlibFileName)
    get_hamiltonian_fortran = fortlib.proxya_get_hamiltonian
    get_density_matrix_fortran = fortlib.proxya_get_density_matrix
except Exception as e:
    fortlib = None
    raise e

__all__ = ["get_hamiltonian_module"]
__all__ = ["get_density_matrix_module"]


def get_hamiltonian_proxy(*args, **kwargs):
    raise NotImplementedError("implement this in an external module!")


def get_hamiltonian_module(eng, coords, atomTypes, symbols, verb):
    if eng.name == "ProxyA":
        hamiltonian = get_hamiltonian_proxy(coords, atomTypes=np.zeros((1), dtype=int), verb=False)

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
            ctypes.c_bool,
        ]
        # Passing a pointer to Fotran
        coords_ptr = coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        atomTypes_ptr = atomTypes.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        hamiltonian = np.zeros((norbs, norbs))
        ham_ptr = hamiltonian.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        err = get_hamiltonian_fortran(
            ctypes.c_int(nats), ctypes.c_int(norbs), coords_ptr, atomTypes_ptr, ham_ptr, ctypes.c_bool(verb)
        )

    return hamiltonian


def get_density_matrix_modules(eng,nocc,hamiltonian,verb=False):

    if(eng.name == "ProxyA"):
        density_matrix = get_densityMatrix(hamiltonian,nocc,verb=False)
    elif(eng.name == "ProxyAFortran"):
        #H needs to be flattened 
        norbs = len(hamiltonian[:,0])
        ht = hamiltonian.T
        #Specify arguments type as a pointers
        get_density_matrix_fortran.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_bool]
        #Passing a pointer to Fortran 
        hamiltonian_ptr = hamiltonian.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        density_matrix = np.zeros((norbs,norbs))
        density_matrix_ptr = density_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        err = get_density_matrix_fortran(
                ctypes.c_int(norbs),ctypes.c_int(nocc),hamiltonian_ptr,density_matrix_ptr,ctypes.c_bool(verb)
        )


    return density_matrix

