import ctypes
import os

import numpy as np
from sedacs.message import * 
import sys 
from sedacs.engine import Engine
from sedacs.globals import *
from sedacs.periodic_table import PeriodicTable 

# import the shared library
try:
    fortlibFileName = os.environ["PROXYA_FORTRAN_PATH"] + "/proxya_fortran.so"
    fortlib = True
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
    #from proxies.python.first_level import get_hamiltonian_proxy
    #from proxies.python.first_level import get_density_matrix_proxy

    #import inspect
    #print(inspect.getfile(get_density_matrix_proxy))
    from proxies.python.hamiltonian import get_hamiltonian_proxy
    from proxies.python.density_matrix import get_density_matrix_proxy
    from proxies.python.evals_dvals import get_evals_dvals_proxy

    #from energy_and_forces import get_ppot_energy_expo_proxy
    # #from energy_and_forces import get_ppot_forces_expo_proxy
    # #from energy_and_forces import get_tb_forces_proxy
    from proxies.python.init_proxy import init_proxy_proxy
    from proxies.python.hamiltonian import build_coul_ham_proxy
except Exception as e:
    pythlib = None
    raise e


__all__ = ["get_hamiltonian_module", "get_density_matrix_module",
        "get_ppot_energy_expo", "get_ppot_forces_expo", "init_proxy",
        "get_tb_forces_module","build_coul_ham_module"
        ]

#Initialize the proxy code
def init_proxy(symbols,orbs):
    init_proxy_proxy(symbols,orbs)
    

def build_coul_ham_module(eng,ham0,vcouls,types,charges,orbital_based,hindex,overlap=None,verb=False):
    if eng.name == "ProxyAPython":
        ham = build_coul_ham_proxy(ham0,vcouls,types,charges,orbital_based,hindex,overlap=overlap,verb=False)
    elif eng.name == "ProxyAFortran":
        error_at("build_coul_ham_module","ProxyAFortran version not implemented yet")
    elif eng.name == "ProxyAC":
        error_at("build_coul_ham_module","ProxyAC version not implemented yet")
    elif eng.name == "LATTE":
        ham = ham0
    else:
        error_at("build_coul_ham_module","No specific engine type defined")
    
    return ham


def get_hamiltonian_module(eng,partIndex,nparts,norbs,latticeVectors,coords,atomTypes,symbols,get_overlap=True,verb=False,newsystem=True, keepmem=False):
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
    elif eng.name == "LATTE":

        # Import the shared library
        latteLibFileName = os.environ['LATTE_PATH'] + '/liblatte.so'

        latteLib = ctypes.CDLL(latteLibFileName)
        latte_compute_f = latteLib.latte_compute 

        #Periodic table: We use this to pass the chemical atom types as integer instead of characters.
        pt = PeriodicTable()
        compflag = 1
        nats = len(coords[:,0])
        ncores = norbs

        #    Getting atomic numbers
        nTypes = len(symbols)
        atomicNumbers = np.zeros((nTypes),dtype=np.int32)
        atomTypes32 = np.zeros((nats),dtype=np.int32)
        atomTypes32[:] = atomTypes
        for i in range(len(symbols)):
            atomicNumbers[i] = pt.get_atomic_number(symbols[i])

        # Vectorizing 2D arrays for C-Fortran interoperability
        coordsFlat_in = np.zeros(3 * nats)  # Vectorized coordinates
        forcesFlat_out = np.zeros(3 * nats)  # Vectorized forces 
        hamFlat_out = np.zeros(norbs * norbs)  # Vectorized hamiltonian
        overFlat_out = np.zeros(norbs * norbs)  # Vectorized overlap
        dmFlat_out = np.zeros(norbs * norbs)  # Vectorized density matrix 
        evalsFlat_out = np.zeros(norbs)  # We call this one Flat just for consistency 
        dvalsFlat_out = np.zeros(norbs)  # Same here 
        chargesFlat_out = np.zeros(nats)  # Same here 
        energyFlat_out = np.zeros(1)  # Same here         

        for i in range(nats):
            coordsFlat_in[3*i] = coords[i,0]
            coordsFlat_in[3*i+1] = coords[i,1]
            coordsFlat_in[3*i+2] = coords[i,2]

        latticeVectorsFlat = np.zeros((9))
        latticeVectorsFlat[0] = latticeVectors[0,0]
        latticeVectorsFlat[1] = latticeVectors[0,1]
        latticeVectorsFlat[2] = latticeVectors[0,2]

        latticeVectorsFlat[3] = latticeVectors[1,0]
        latticeVectorsFlat[4] = latticeVectors[1,1]
        latticeVectorsFlat[5] = latticeVectors[1,2]

        latticeVectorsFlat[6] = latticeVectors[2,0]
        latticeVectorsFlat[7] = latticeVectors[2,1]
        latticeVectorsFlat[8] = latticeVectors[2,2]
        
        vcoulsFlat = np.zeros(nats)

        # Inputs
        coords_ptr = coordsFlat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        atomTypes_ptr = atomTypes32.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        atomicNumbers_ptr = atomicNumbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        latticeVectors_ptr = latticeVectorsFlat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        vcouls_ptr = vcoulsFlat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Outputs
        ham_ptr = hamFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        over_ptr = overFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dm_ptr = dmFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        charges_ptr = chargesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        evals_ptr = evalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dvals_ptr = dvalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        energy_ptr = energyFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        forces_ptr = forcesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # Call to the fortran funtion
        err = latte_compute_f(
            ctypes.c_int(compflag),
            ctypes.c_int(partIndex+1),
            ctypes.c_int(nparts),
            ctypes.c_int(norbs),
            ctypes.c_int(ncores),
            ctypes.c_int(nats),
            ctypes.c_int(nats),
            ctypes.c_int(nTypes),
            ctypes.c_double(0.0),
            vcouls_ptr,
            coords_ptr,
            latticeVectors_ptr,
            atomTypes_ptr,
            atomicNumbers_ptr,
            ham_ptr,
            over_ptr,
            dm_ptr,
            charges_ptr,
            evals_ptr,
            dvals_ptr,
            energy_ptr,
            forces_ptr,
            ctypes.c_int(verb),
            ctypes.c_int(newsystem),
            ctypes.c_int(keepmem),
        )

        # Back to a 2D array for the forces
        hamiltonian = np.zeros((norbs, norbs))
        overlap = np.zeros((norbs, norbs))

        for i in range(norbs):
            hamiltonian[:, i] = hamFlat_out[norbs * i : norbs + norbs * i]
            overlap[:, i] = overFlat_out[norbs * i : norbs + norbs * i]

    else:
        error_at("get_hamiltonian_module","No specific engine type defined")

    if(get_overlap):
        return hamiltonian, overlap
    else:
        return hamiltonian


def get_evals_dvals_modules(eng,partIndex,nparts,latticeVectors,coords,atomTypes,symbols,hamiltonian,vcouls,nocc,norbsInCore=None,method="Diag",accel="No",mu=None,etemp=0.0,overlap=None,full_data=False,verb=False,newsystem=True):
    if eng.name == "ProxyAPython":
        evals, dvals = get_evals_dvals_proxy(hamiltonian, nocc, norbsInCore=norbsInCore, method="Diag", accel="No", mu=mu, etemp=etemp, overlap=overlap, full_data=full_data, verb=verb)

    elif eng.name == "ProxyAFortran":
        error_at("get_evals_dvals_modules","Not implemented yet.")

    elif eng.name == "LATTE":

        # Import the shared library
        latteLibFileName = os.environ['LATTE_PATH'] + '/liblatte.so'

        latteLib = ctypes.CDLL(latteLibFileName)
        latte_compute_f = latteLib.latte_compute 

        #Periodic table: We use this to pass the chemical atom types as integer instead of characters.
        pt = PeriodicTable()
        compflag = 2
        nats = len(coords[:,0])
        norbs = len(hamiltonian[:,0])
        keepmem = False

        #    Getting atomic numbers
        nTypes = len(symbols)
        atomicNumbers = np.zeros((nTypes),dtype=np.int32)
        atomTypes32 = np.zeros((nats),dtype=np.int32)
        atomTypes32[:] = atomTypes 
        for i in range(len(symbols)):
            atomicNumbers[i] = pt.get_atomic_number(symbols[i])

        # Vectorizing 2D arrays for C-Fortran interoperability
        coordsFlat_in = np.zeros(3 * nats)  # Vectorized coordinates
        forcesFlat_out = np.zeros(3 * nats)  # Vectorized forces 
        hamFlat_out = np.zeros(norbs * norbs)  # Vectorized hamiltonian
        overFlat_out = np.zeros(norbs * norbs)  # Vectorized overlap
        dmFlat_out = np.zeros(norbs * norbs)  # Vectorized density matrix 
        evalsFlat_out = np.zeros(norbs)  # We call this one Flat just for consistency 
        dvalsFlat_out = np.zeros(norbs)  # Same here 
        chargesFlat_out = np.zeros(nats)  # Same here 
        energyFlat_out = np.zeros(1)  # Same here         

        for i in range(nats):
            coordsFlat_in[3*i] = coords[i,0]
            coordsFlat_in[3*i+1] = coords[i,1]
            coordsFlat_in[3*i+2] = coords[i,2]
        
        latticeVectorsFlat = np.zeros((9))
        latticeVectorsFlat[0] = latticeVectors[0,0]
        latticeVectorsFlat[1] = latticeVectors[0,1]
        latticeVectorsFlat[2] = latticeVectors[0,2]

        latticeVectorsFlat[3] = latticeVectors[1,0]
        latticeVectorsFlat[4] = latticeVectors[1,1]
        latticeVectorsFlat[5] = latticeVectors[1,2]

        latticeVectorsFlat[6] = latticeVectors[2,0]
        latticeVectorsFlat[7] = latticeVectors[2,1]
        latticeVectorsFlat[8] = latticeVectors[2,2]
        
        # Inputs
        coords_ptr = coordsFlat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        atomTypes_ptr = atomTypes32.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        atomicNumbers_ptr = atomicNumbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        latticeVectors_ptr = latticeVectorsFlat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        vcouls_ptr = vcouls.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Outputs
        ham_ptr = hamFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        over_ptr = overFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dm_ptr = dmFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        charges_ptr = chargesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        evals_ptr = evalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dvals_ptr = dvalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        energy_ptr = energyFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        forces_ptr = forcesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Call to the fortran funtion
        err = latte_compute_f(
            ctypes.c_int(compflag),
            ctypes.c_int(partIndex+1),
            ctypes.c_int(nparts),
            ctypes.c_int(norbs),
            ctypes.c_int(int(norbsInCore)),
            ctypes.c_int(nats),
            ctypes.c_int(nats),
            ctypes.c_int(nTypes),
            ctypes.c_double(mu),
            vcouls_ptr,
            coords_ptr,
            latticeVectors_ptr,
            atomTypes_ptr,
            atomicNumbers_ptr,
            ham_ptr,
            over_ptr,
            dm_ptr,
            charges_ptr,
            evals_ptr,
            dvals_ptr,
            energy_ptr,
            forces_ptr,
            ctypes.c_int(verb),
            ctypes.c_int(newsystem),
            ctypes.c_int(keepmem),
        )

        #Back to a 2D array for the forces
        evals = np.zeros((norbs))
        dvals = np.zeros((norbs))
        evals[:] = evalsFlat_out[:]
        dvals[:] = dvalsFlat_out[:]

    return evals, dvals


def get_density_matrix_modules(eng,partIndex,nparts,norbs,latticeVectors,coords,atomTypes,symbols,hamiltonian,vcouls,nocc,norbsInCore=None,method="Diag",accel="No",mu=None,etemp=0.0,overlap=None,full_data=False,verb=False,newsystem=True,keepmem=False):
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

    elif eng.name == "LATTE":

        # Import the shared library
        latteLibFileName = os.environ['LATTE_PATH'] + '/liblatte.so'

        latteLib = ctypes.CDLL(latteLibFileName)
        latte_compute_f = latteLib.latte_compute 

        #Periodic table: We use this to pass the chemical atom types as integer instead of characters.
        pt = PeriodicTable()
        compflag = 3
        nats = len(coords[:,0])

        #    Getting atomic numbers
        nTypes = len(symbols)
        atomicNumbers = np.zeros((nTypes),dtype=np.int32)
        atomTypes32 = np.zeros((nats),dtype=np.int32)
        atomTypes32[:] = atomTypes 
        for i in range(len(symbols)):
            atomicNumbers[i] = pt.get_atomic_number(symbols[i])

        # Vectorizing 2D arrays for C-Fortran interoperability
        coordsFlat_in = np.zeros(3 * nats)  # Vectorized coordinates
        forcesFlat_out = np.zeros(3 * nats)  # Vectorized forces 
        hamFlat_out = np.zeros(norbs * norbs)  # Vectorized hamiltonian
        overFlat_out = np.zeros(norbs * norbs)  # Vectorized overlap
        dmFlat_out = np.zeros(norbs * norbs)  # Vectorized density matrix 
        evalsFlat_out = np.zeros(norbs)  # We call this one Flat just for consistency 
        dvalsFlat_out = np.zeros(norbs)  # Same here 
        chargesFlat_out = np.zeros(nats)  # Same here 
        energyFlat_out = np.zeros(1)  # Same here         

        for i in range(nats):
            coordsFlat_in[3*i] = coords[i,0]
            coordsFlat_in[3*i+1] = coords[i,1]
            coordsFlat_in[3*i+2] = coords[i,2]
        
        latticeVectorsFlat = np.zeros((9))
        latticeVectorsFlat[0] = latticeVectors[0,0]
        latticeVectorsFlat[1] = latticeVectors[0,1]
        latticeVectorsFlat[2] = latticeVectors[0,2]

        latticeVectorsFlat[3] = latticeVectors[1,0]
        latticeVectorsFlat[4] = latticeVectors[1,1]
        latticeVectorsFlat[5] = latticeVectors[1,2]

        latticeVectorsFlat[6] = latticeVectors[2,0]
        latticeVectorsFlat[7] = latticeVectors[2,1]
        latticeVectorsFlat[8] = latticeVectors[2,2]
        
        # Inputs
        coords_ptr = coordsFlat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        atomTypes_ptr = atomTypes32.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        atomicNumbers_ptr = atomicNumbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        latticeVectors_ptr = latticeVectorsFlat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        vcouls_ptr = vcouls.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Outputs
        ham_ptr = hamFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        over_ptr = overFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dm_ptr = dmFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        charges_ptr = chargesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        evals_ptr = evalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dvals_ptr = dvalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        energy_ptr = energyFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        forces_ptr = forcesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		
 
        # Call to the fortran funtion
        err = latte_compute_f(
            ctypes.c_int(compflag),
            ctypes.c_int(partIndex+1),
            ctypes.c_int(nparts),
            ctypes.c_int(norbs),
            ctypes.c_int(int(norbsInCore)),
            ctypes.c_int(nats),
            ctypes.c_int(nats),
            ctypes.c_int(nTypes),
            ctypes.c_double(mu),
            vcouls_ptr,
            coords_ptr,
            latticeVectors_ptr,
            atomTypes_ptr,
            atomicNumbers_ptr,
            ham_ptr,
            over_ptr,
            dm_ptr,
            charges_ptr,
            evals_ptr,
            dvals_ptr,
            energy_ptr,
            forces_ptr,
            ctypes.c_int(verb),
            ctypes.c_int(newsystem),
            ctypes.c_int(keepmem),
        )


        #Back to a 2D array for the forces
        density_matrix = np.zeros((norbs, norbs))
        charges = np.zeros((nats))
        charges[:] = chargesFlat_out[:] 

        for i in range(norbs):
            density_matrix[:, i] = dmFlat_out[norbs * i : norbs + norbs * i]

        return density_matrix, charges

    else:
        method = eng.method
        accel = eng.accel
        if(full_data):
            density_matrix,evals,dvals = get_density_matrix_proxy(hamiltonian,nocc,norbsInCore=None,method=method,accel=accel,mu=mu, overlap=overlap,full_data=full_data,verb=False)
        else:
            density_matrix = get_density_matrix_proxy(hamiltonian,nocc,norbsInCore=None,method=method,accel=accel,mu=mu,overlap=overlap,full_data=full_data,verb=False)
#        error_at("get_density_matrix_module","No specific engine type defined")

    if(full_data):
        return density_matrix,evals,dvals
    else:
        return density_matrix

def get_energy_forces_modules(eng,partIndex,nparts,norbs,hamiltonian,latticeVectors,coords,atomTypes,symbols,vcouls,nocc,norbsInCore=None,numberOfCoreAtoms=None,mu=None,etemp=0.0,verb=False,newsystem=True,keepmem=False):
    if eng.name == "ProxyAPython":
        error_at("get_energy_force_modules","Not implemented yet.")

    elif eng.name == "ProxyAFortran":
        error_at("get_energy_force_modules","Not implemented yet.")

    elif eng.name == "LATTE":
        
        # Import the shared library
        latteLibFileName = os.environ['LATTE_PATH'] + '/liblatte.so'

        latteLib = ctypes.CDLL(latteLibFileName)
        latte_compute_f = latteLib.latte_compute 

        #Periodic table: We use this to pass the chemical atom types as integer instead of characters.
        pt = PeriodicTable()
        compflag = 4
        nats = len(coords[:,0])

        #    Getting atomic numbers
        nTypes = len(symbols)
        atomicNumbers = np.zeros((nTypes),dtype=np.int32)
        atomTypes32 = np.zeros((nats),dtype=np.int32)
        atomTypes32[:] = atomTypes 
        for i in range(len(symbols)):
            atomicNumbers[i] = pt.get_atomic_number(symbols[i])

        # Vectorizing 2D arrays for C-Fortran interoperability
        coordsFlat_in = np.zeros(3 * nats)  # Vectorized coordinates
        forcesFlat_out = np.zeros(3 * nats)  # Vectorized forces 
        hamFlat_out = np.zeros(norbs * norbs)  # Vectorized hamiltonian
        overFlat_out = np.zeros(norbs * norbs)  # Vectorized overlap
        dmFlat_out = np.zeros(norbs * norbs)  # Vectorized density matrix 
        evalsFlat_out = np.zeros(norbs)  # We call this one Flat just for consistency 
        dvalsFlat_out = np.zeros(norbs)  # Same here 
        chargesFlat_out = np.zeros(nats)  # Same here 
        energyFlat_out = np.zeros(1)  # Same here         

        for i in range(nats):
            coordsFlat_in[3*i] = coords[i,0]
            coordsFlat_in[3*i+1] = coords[i,1]
            coordsFlat_in[3*i+2] = coords[i,2]
        
        latticeVectorsFlat = np.zeros((9))
        latticeVectorsFlat[0] = latticeVectors[0,0]
        latticeVectorsFlat[1] = latticeVectors[0,1]
        latticeVectorsFlat[2] = latticeVectors[0,2]

        latticeVectorsFlat[3] = latticeVectors[1,0]
        latticeVectorsFlat[4] = latticeVectors[1,1]
        latticeVectorsFlat[5] = latticeVectors[1,2]

        latticeVectorsFlat[6] = latticeVectors[2,0]
        latticeVectorsFlat[7] = latticeVectors[2,1]
        latticeVectorsFlat[8] = latticeVectors[2,2]
        
        # Inputs
        coords_ptr = coordsFlat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        atomTypes_ptr = atomTypes32.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        atomicNumbers_ptr = atomicNumbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        latticeVectors_ptr = latticeVectorsFlat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        vcouls_ptr = vcouls.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Outputs
        ham_ptr = hamFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        over_ptr = overFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dm_ptr = dmFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        charges_ptr = chargesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        evals_ptr = evalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dvals_ptr = dvalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        energy_ptr = energyFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        forces_ptr = forcesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    
        # Call to the fortran funtion
        err = latte_compute_f(
            ctypes.c_int(compflag),
            ctypes.c_int(partIndex+1),
            ctypes.c_int(nparts),
            ctypes.c_int(norbs),
            ctypes.c_int(norbsInCore),
            ctypes.c_int(numberOfCoreAtoms),
            ctypes.c_int(nats),
            ctypes.c_int(nTypes),
            ctypes.c_double(mu),
            vcouls_ptr,
            coords_ptr,
            latticeVectors_ptr,
            atomTypes_ptr,
            atomicNumbers_ptr,
            ham_ptr,
            over_ptr,
            dm_ptr,
            charges_ptr,
            evals_ptr,
            dvals_ptr,
            energy_ptr,
            forces_ptr,
            ctypes.c_int(verb),
            ctypes.c_int(newsystem),
            ctypes.c_int(keepmem),
        )

        # Back to a 2D array for the forces
        forces = np.zeros((nats, 3))
        for i in range(nats):
            forces[i, 0] = forcesFlat_out[i * 3 + 0]
            forces[i, 1] = forcesFlat_out[i * 3 + 1]
            forces[i, 2] = forcesFlat_out[i * 3 + 2]

    else:
        error_at("get_energy_force_modules","No specific engine type defined")

    return energyFlat_out[0], forces 

def get_ppot_energy_expo(coords,types):

    energy = get_ppot_energy_expo_proxy(coords,types)

    return energy


def get_ppot_forces_expo(coords,types):

    forces = get_ppot_forces_expo_proxy(coords,types) 

    return forces


def get_tb_forces_module(ham,rho,charges,field,coords,atomTypes,symbols,overlap=None,verb=False):

    forces = get_tb_forces_proxy(ham,rho,charges,field,coords,atomTypes,symbols,overlap=None,verb=False)

    return forces



def call_latte_modules(eng,Sy,verb=False,newsystem=True):

    if eng.name == "LATTE":

        coords = Sy.coords
        latticeVectors = Sy.latticeVectors
        symbols = np.array(Sy.symbols)[Sy.types] 
        types = Sy.types

        # Import the shared library
        latteLibFileName = os.environ['LATTE_PATH'] + '/liblatte.so'

        latteLib = ctypes.CDLL(latteLibFileName)
        latte_compute_f = latteLib.latte_c_bind

        #Periodic table: We use this to pass the chemical atom types as integer instead of characters.
        pt = PeriodicTable()
        compflag = np.zeros(5) 
        nats = len(coords[:,0])
        norbs = Sy.norbs
        err = True 

        #    Getting atomic numbers
        #nTypes = len(symbols)
        nTypes = len(Sy.symbols)
        atomTypes32 = np.zeros((nats),dtype=np.int32)
        atomTypes32[:] = Sy.types + 1 
        #masses = np.where(symbols == 'H', 1.0, 0.0) + np.where(symbols == 'O', 16.0, 0.0)
        masses = np.zeros(len(Sy.symbols),dtype=np.float64)
        for i in range(len(Sy.symbols)):
            masses[i] = pt.mass[ pt.get_atomic_number(Sy.symbols[i]) ]

        # Vectorizing 2D arrays for C-Fortran interoperability
        coordsFlat_in = np.zeros(3 * nats)  # Vectorized coordinates
        forcesFlat_out = np.zeros((3, nats), order='F')  # Vectorized forces 
        chargesFlat_out = np.zeros(nats)  # Same here 
        velFlat_out = np.zeros((3, nats), order='F')
        energyFlat_out = np.zeros(1)
        virialFlat_out = np.zeros((6,), order='F')
#        coords = coords.T
#        coords = np.asfortranarray(coords)
        for i in range(nats):
            coordsFlat_in[3*i] = coords[i,0]
            coordsFlat_in[3*i+1] = coords[i,1]
            coordsFlat_in[3*i+2] = coords[i,2]
        
        
#        latticeVectorsFlat = np.zeros((9))
        xlo = np.zeros(3) 
        xhi = np.zeros(3) 
        xhi[0] = latticeVectors[0,0]
        xhi[1] = latticeVectors[1,1]
        xhi[2] = latticeVectors[2,2]
        #xlo[:] = -100
        #xhi[:] = 100
        xy, xz, yz = 0.0, 0.0, 0.0
        maxiter = -1

        # Inputs
        coords_ptr = coordsFlat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #coords_ptr = coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        atomTypes_ptr = atomTypes32.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        xlo_ptr = xlo.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        xhi_ptr = xhi.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        masses_ptr = masses.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        compflag_ptr = compflag.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        vel_ptr = velFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Outputs
        charges_ptr = chargesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        forces_ptr = forcesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        energy_ptr = energyFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        virial_ptr = virialFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # Call to the fortran funtion
        latte_compute_f(
            compflag_ptr,
            ctypes.c_int(nats),
            coords_ptr,
            atomTypes_ptr,
            ctypes.c_int(nTypes),
            masses_ptr,
            xlo_ptr,
            xhi_ptr,
            ctypes.c_double(xy),
            ctypes.c_double(xz),
            ctypes.c_double(yz),
            forces_ptr,
            ctypes.c_int(maxiter),
            energy_ptr,
            vel_ptr,
            ctypes.c_double(0.5),
            virial_ptr,
            charges_ptr,
            ctypes.c_int(1),
            ctypes.c_bool(err),
        )


        #Back to a 2D array for the forces
        charges = np.zeros((nats))
        charges[:] = chargesFlat_out[:] 

        return charges

    else:
        error_at("call_latte_module","Not implemented yet")
