from proxy_a import *
import ctypes as ct
import os 
# import the shared library
fortlibFileName = os.environ['PROXYA_FORTRAN_PATH'] + '/proxya_fortran.so'

try:
    fortlib = ct.CDLL(fortlibFileName) 
    f = fortlib.proxya_get_hamiltonian
except:
    fortlib = None

def sdc_get_hamiltonian_module(eng,coords,atomTypes,symbols,verb):
    
    if(eng.name == "ProxyA"):
        ham = proxyA_get_hamiltonian(coords,atomTypes=np.zeros((1),dtype=int),verb=False)

    elif(eng.name == "ProxyAFortran"):
        nats = len(coords[:,0])
        norbs = nats

        coords_in = np.zeros(3*nats) #Vectorized coordinates
        for i in range(nats):
            coords_in[3*i] = coords[i,0]
            coords_in[3*i+1] = coords[i,1]
            coords_in[3*i+2] = coords[i,2]

        #Specify arguments type as a pointers
        f.argtypes=[ct.c_int,ct.c_int,ct.POINTER(ct.c_double),ct.POINTER(ct.c_int),ct.POINTER(ct.c_double),ct.c_bool]
        #Passing a pointer to Fotran 
        coords_ptr = coords.ctypes.data_as(ct.POINTER(ct.c_double))
        atomTypes_ptr = atomTypes.ctypes.data_as(ct.POINTER(ct.c_int))
        ham = np.zeros((norbs,norbs))
        ham_ptr = ham.ctypes.data_as(ct.POINTER(ct.c_double))

        err = f(ct.c_int(nats),ct.c_int(norbs),coords_ptr,atomTypes_ptr,ham_ptr,ct.c_bool(verb))
    
    return ham

