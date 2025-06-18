"""density_matrix 
Computes the Density matrix from a given Hamiltonian
This code is only used to guide implemetations and understand which are the 
basic elements needed to interface with the sedacs driver.
"""

import os
import sys

import numpy as np
import scipy.linalg as sp
from hamiltonian_elements import *  
from dnnprt import *
from proxy_global import *
from coordinates import get_random_coordinates
from hamiltonian_random import get_random_hamiltonian
from hamiltonian import get_hamiltonian_proxy
from chemical_potential import fermi_dirac, get_mu
from nonortho import get_xmat
import gpuLibInterface as gpu
from init_proxy import init_proxy_accelerators
from proxy_global import bring_ham_list, bring_dm_list, bring_cublas_handle_list, bring_stream_list

try:
    import ctypes


    gpuLib = True
    arch = "nvda"
    pwd = os.getcwd()

    if arch == "nvda":
        print("loading nvidia...")
        lib = ctypes.CDLL("/home/finkeljo/sedacs-gpmdsp2/src/sedacs/gpu/nvda/libnvda.so")
    if arch == "amd":
        lib = ctypes.CDLL(str((src_path() / "gpu/amd/libamd.so").absolute()))

except:
    gpuLib = False

import ctypes



__all__ = [
    "get_density_matrix_proxy",
    "get_density_matrix_gpu",
]


## Computes the Density matrix from a given Hamiltonian.
# @author Anders Niklasson
# @brief This will create a Density matrix \f$ \rho \f$
# \f[ \rho  =  \sum^{nocc} v_k v_k^T \f]
# where \f$ v_k \f$ are the eigenvectors of the matrix \f$ H \f$
#
# @param ham Hamiltonian matrix
# @param nocc Number of occupied orbitals
# @param core_size Number of atoms in the cores.
# @param method Type of algorithm used to compute DM
# @param accel Type of accelerator/special device used to compute DM. Default is No and
# will only use numpy.
# @param mu Chemical potential. If set to none, the calculation will use nocc
# @param etemp Electronic temperature
# @param overlap Overlap matrix
# @param verb Verbosity. If True is passed, information is printed.
#
# @return rho Density matrix
#
def get_density_matrix_proxy(ham, nocc, norbsInCore=None, method="Diag", accel="No", mu=None, etemp=0.0, overlap=None, full_data=False, verb=False, lib=None):
    """Calcualtion of the full density matrix from H"""
    if verb:
        print("Computing the Density matrix")

    norbs = len(ham[:, 0])
    ham_orth = np.zeros((norbs, norbs))
    if overlap is not None:
        # Get the inverse overlap factor
        zmat = get_xmat(overlap, method="Diag", accel="No", verb=False)

        # Orthogonalize Hamiltonian
        ham_orth = np.matmul(np.matmul(np.transpose(zmat), ham), zmat)
    else:
        ham_orth[:, :] = ham[:, :]

    if method == "Diag" and accel == "No":
        evals, evects = sp.eigh(ham_orth)
        print("evals",evals)
        homoIndex = nocc - 1
        lumoIndex = nocc 
         
        #If mu is not set we set mu HOMO+LUMO/2
        if (mu is None):
            mu = 0.5 * (evals[homoIndex] + evals[lumoIndex])
        else:
            pass

        if verb:
            print("Chemical potential = ", mu)

        rho = np.zeros((norbs, norbs))
        if verb:
            print("Eigenvalues of H:", evals)

        #If the electronic temperature is 0
        if(etemp < 1.0E-10):
            for i in range(norbs):
                if evals[i] < mu:
                    rho = rho + np.outer(evects[:, i], evects[:, i])
        else:
            #mu = get_mu(mu, evals, etemp, nocc, dvals=None, kB=8.61739e-5, verb=False)
            fvals = np.zeros((norbs))
            fvals = fermi_dirac(mu, evals, etemp, kB=8.61739e-5) 
            for i in range(norbs):
                if evals[i] < mu:
                    rho = rho + fvals[i]*np.outer(evects[:, i], evects[:, i])

    elif method == "SP2" and accel == "No":
        #rho = dnnprt(ham_orth, norbs, nocc, H1=None, refi=False)
        #rho = movingmu_sp2(ham,mu=mu,thresh=0.0,miniter=5,maxiter=50,sp2conv=1.0E-6,idemtol=1.0E-6,verb=True)
        #rho = golden_sp2(ham,mu=mu,thresh=0.0,miniter=5,maxiter=50,sp2conv=1.0E-6,idemtol=1.0E-6,verb=True)
        rho = sp2_basic(ham,nocc,thresh=0.0,minsp2iter=5,maxsp2iter=30,sp2conv=1.0E-5,idemtol=1.0E-5,verb=True)
    
    elif method == "SP2" and accel == "TC":

        accel_lib = bring_accel_lib()
    
        #print("--------set stream in python--------------")
        #stream=gpu.set_stream(accel_lib); 
        #print(id(stream),stream)
        


        #print("\n")
        #print("--------init cublas handle in python--------------")
        #cublas_handle = gpu.cublasInit(accel_lib)
        #print(id(cublas_handle),cublas_handle)
        
        cublas_handle_list=bring_cublas_handle_list()
        streams_list  = bring_stream_list()
        dev_list      = bring_dev_list()
        d_ham = dev_list[0] 
        d_dm  = dev_list[1] 

        test_handle = cublas_handle_list[0]
        test_stream = streams_list[0]

        # determine size
        size_of_double = 8 #bytes
        matSize  = norbs * norbs * size_of_double

        size_of_float  = 4 #bytes
        matSize_f = norbs * norbs * size_of_float

        pinned_ham = dev_list[10]
        pinned_dm = dev_list[11]
        #gpu.memcpyHtoH(pinned_ham, ham, matSize, accel_lib)
        #gpu.memcpyHtoD(d_ham, pinned_ham, matSize, accel_lib)
        gpu.memcpyHtoD(d_ham, ham, matSize, accel_lib)
        #e,v = np.linalg.eigh(ham)
        #print(e)
        rho=np.empty((norbs,norbs))
        print("device is = ", gpu.get_device(accel_lib))
        #ham = d_ham_list[0]
        #dm = d_dm_list[0]


        gpu.dmDNNSP2(dev_list,norbs,nocc,test_handle,test_stream,accel_lib)

        #gpu.dmGoldenSP2(d_ham_list[0],d_dm_list[0],norbs,mu,cublas_handle,accel_lib)
        gpu.memcpyDtoH(rho, d_dm, matSize, accel_lib)
        #gpu.memcpyDtoH(pinned_dm, d_dm_list[0], matSize, accel_lib)
        #gpu.memcpyDtoH(pinned_dm, d_dm, matSize, accel_lib)
        #gpu.memcpyHpinnedtoH(rho, pinned_dm, matSize, accel_lib)
        #e1,v1 = np.linalg.eigh(rho)

        #gpu.dmMovingMuSP2(d_ham_list[0],d_dm_list[0],norbs,mu,cublas_handle,accel_lib)
        #gpu.memcpyDtoH(rho, d_dm_list[0], matSize, accel_lib)
        #e2,v2 = np.linalg.eigh(rho)
        #for i in range(0,norbs):
        #    print(e1[i],e2[i])
        #exit()
        #rho = sp2_basic(ham,nocc,thresh=0.0,minsp2iter=5,maxsp2iter=30,sp2conv=1.0E-5,idemtol=1.0E-5,verb=True)
        
    elif method == "SP2" and accel == "PBML":
        print("No method yet")
    else:
        print("The combination of method and accelerator is unknown")
        exit(0)

    if(overlap is not None):
        rho = np.matmul(np.matmul(zmat,rho),np.transpose(zmat)) 

    print(norbs)
    dvals = np.zeros((norbs))
    if(method == "Diag"):
        if (overlap is not None):
            overTimesEvects = np.dot(overlap,evects)
        else:
            overTimesEvects = evects
        for i in range(norbs):
            #dvals = np.append(dvals, np.inner(evects[:norbsInCore,i],overTimesEvects[:norbsInCore,i]))
            dvals[i] = np.inner(evects[:norbsInCore,i],overTimesEvects[:norbsInCore,i])
    else:
        if(norbsInCore is not None):
            dvals[:] = norbsInCore/norbs
        else:
            dvals[:] = 1.0

        evals = np.zeros(norbs)
        evals = np.diag(ham_orth) #We estimate the eigenvaluse from the Girshgorin centers


    #print("Ham\n",ham_orth)
    #print("Mu",mu)
    #print("DM\n")
    diagonal = np.diag(rho)
    #print("tr",np.trace(rho))
    #print("NOCC",nocc,norbs)
    #for i in range(norbs):
    #    print(diagonal[i])

    evals, evects = sp.eigh(rho)
    if(full_data):
        return rho, evals, dvals
    else:
        return rho



## Computes the Density matrix from a given Hamiltonian.
# @author Josh Finkelstein
# @brief This will create a "zero-temperature" Density matrix \f$ \rho \f$
# \f[ \rho  =  \sum^{nocc} v_k v_k^T \f]
# where \f$ v_k \f$ are the eigenvectors of the matrix \f$ H \f$
# using GPU/AI accelerator library
#
# @param H Hamiltonian matrix
# @param Nocc Number of occupied orbitals
# @param verb Verbosity. If True is passed, information is printed.
#
# @return D Density matrix
#
def get_density_matrix_gpu(H, N, Nocc, lib, verb=False):
    """Calcualted the full density matrix from H"""
    if verb:
        print("Computing the Density matrix using GPU/AI accel library")

    # init DM
    D = np.zeros((N, N))
    kbt = 0.1

    # get DM from cusolver diag
    #dm = gpu.dmDNNSP2(H,D,N,Nocc,lib)
    # dm = gpu.dmCheby(H,D,N,Nocc,kbt,lib)
    print("Density matrix=", D)
    # dm = gpu.dmDiag(H,D,N,Nocc,kbt,lib)
    # print("Density matrix=",dm)
    #dm = gpu.dmMLSP2(H, D, N, Nocc, lib)
    return D


if __name__ == "__main__":
    n = len(sys.argv)
    if n == 1:
        print("Give the name of the algorithm and the total number of atoms. Example:\n")
        print("density_matrix get_density_matrix_proxy 100\n")
        sys.exit(0)
    else:
        algo = str(sys.argv[1])
        nats = int(sys.argv[2])

    verb = True
    coords = get_random_coordinates(nats)
    atomTypes = np.zeros((nats),dtype=int)
    symbols = []*nats
    symbols[:] = "H"

    filename = "aosa_parameters.dat"
    bas_per_atom = [1]
    tbparams = read_tbparams(filename, symbols, bas_per_atom)

    nvtx.push_range("get hamiltonian proxy",color="blue", domain="get proxy h")
    ham, over = get_hamiltonian_proxy(coords, atomTypes, symbols, get_overlap=True)
    nvtx.pop_range(domain="get proxy h")
    if (gpuLib == True):
        ##
        size_of_double = 8  # bytes
        matSize = nats * nats * size_of_double
        #        cublas_handle = gpu.cublasInit(lib)

        #if (eng.accel == "TC"):
        init_proxy_accelerators(1,4096)

        ## async copy of ham from host to device
        gpu.memcpyHtoD(dev_list[0], ham, matSize, lib)
         
    



    if(algo == "get_density_matrix_proxy"):
        occ = int(float(nats) / 2.0) 
        rho1 = get_density_matrix_proxy(ham, occ, method="SP2")
        print(lib)
        rho2 = get_density_matrix_proxy(ham, occ, method="SP2", accel="TC",d_ham=d_ham,d_dm=d_dm,lib=lib)
        print("Density matrix=", rho1)
        print("Density matrix=", rho2)

