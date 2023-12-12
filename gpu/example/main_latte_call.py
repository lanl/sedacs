#!/usr/bin/env python3

import numpy as np
#from readtb import *
from latte import *
from coordinates import *
from ptable import *
import time

# to call cuda/hip
import numpy.ctypeslib as npct
import ctypes 


def call_nlist(x,y,z,nlist,num_atoms):
    size = 32*num_atoms
    array_type1 = ctypes.c_double*num_atoms     
    array_type2 = ctypes.c_int*size 
    x_c = array_type1(*x)                       
    y_c = array_type1(*y)                       
    z_c = array_type1(*z)                       
    nlist_c = array_type2(*nlist)               
    rcut = ctypes.c_float(2.0)
    nats = ctypes.c_int(num_atoms)
    dev = ctypes.c_int(0)
    print("dev=",dev) 
    tic = time.perf_counter()
    lib.nlist(x_c,y_c,z_c,nlist_c,rcut,nats,0)          
    toc = time.perf_counter()
    print(f"Time = {toc - tic:0.4f} seconds")
    return list(nlist)                          

def test_interface(in1,in2,K,M,N):
    C_K=ctypes.c_int(K)
    C_M=ctypes.c_int(M)
    C_N=ctypes.c_int(N)
    array_type = ctypes.c_double*N              # equiv. to C double[N] type
    arr1 = array_type(*in1)                     # equiv. to double arr1[N] = {...} instance
    arr2 = array_type(*in2)                     # equiv. to double arr2[N] = {...} instance
    lib.density_mat(arr1,arr2,C_K,C_M,600)      # pointer to array passed to function and modified
    return list(arr2)                           # extract Python floats from ctypes-wrapped array

## General LATTE dm API call 
# This function will take coordinates and atom type and 
# retreive the density matrix. 
# @param box Lattice vectors. box[0,:] = first lattice vectors
# @param symbols List of elements symbols for each atom type
# @param types A list of types for every atom in the system
# @param coords Positions for every atom in the system. coords[0,2] z-coordinate of atom 0
# @return dm Density matrix
#
def get_latte_dm(box,symbols,types,coords):
    nats = len(types) #Number of atoms
    atele = []
    for i in range(nats): atele.append(symbols[types[i]]) #Element for every atom in the system

    #Read bond integrals and atomic info
    noelem,ele,ele1,ele2,basis,hes,hep,hed,hef,hcut,scut,noint,btype, \
        tabh,tabs,sspl,hspl,lentabint,tabr,atocc,mass,hubbardu = \
        read_bondints("electrons.dat","bondints.table",myVerb)

    #Get element pointer
    elempointer = get_elempointer(atele,ele,myVerb)

    #Get the dimension of Hamiltonian
    hdim = get_hdim(nats,elempointer,basis,myVerb)

    #et cutoff list
    cutoffList = get_cutoffList(nats,atele,ele1,ele2,hcut,scut,noint,myVerb)

    #Get integral map
    iglMap = build_integralMap(noint,btype,atele,ele1,ele2,nats,myVerb)

    #Construct the Hamiltonian and Overlap
    smat,ham = build_HS(nats,coords,box,elempointer,basis,\
        hes,hep,hed,hef,hdim,cutoffList,iglMap,\
        tabh,tabs,sspl,hspl,lentabint,tabr,hcut,scut,myVerb)

    #Get the inverse overlap factors
    zmat = genX(smat,method="Diag",verbose=True)

    #Initializing a periodic table
    pt = ptable()

    #Getting number of electrons
    numel = 0
    for i in range(nats):
        atnum = pt.get_atomic_number(symbols[types[i]])
        numel = numel + pt.numel[atnum]

    #Getting the number of occupied states
    nocc = int(numel/2.0)

    #Getting the density matrix
    T = 100.0
    dm,focc,C,e,mu0,entropy = fermi_exp(ham,zmat,nocc,T)

    # cast to to 1d array
    hamm=np.ones((600*600,1))
    for i in range(0,600):
        for j in range(0,600):
            hamm[i+600*j] = ham[i,j]        

    dmm=np.zeros((600*600,1))
  
    K=10
    M=10

    z=funct(hamm,dmm,10,10,600*600)
    
    return ham,z

if(__name__ == '__main__'):

    # Import module
    #pwd="/vast/home/finkeljo/ctypes/simplest/generalcodes/py-to-cpp/"
    #lib = ctypes.CDLL(str(pwd)+"libpymodule-amd.so")
    #lib.density_mat.argtypes = ctypes.POINTER(ctypes.c_double),


    # Import module
    pwd="/vast/home/finkeljo/ctypes/simplest/generalcodes/py-to-cpp/"
    lib = ctypes.CDLL(str(pwd)+"nlist.so")
    lib.nlist.argtypes = ctypes.POINTER(ctypes.c_double),


    myVerb = True
    #Read coordinates from pdb file 
    box,symbols,types,coords = read_pdb_file("coords_1032.pdb",lib="None",verb=myVerb)
 
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]
    
    nlist = np.zeros((1032*32,1))

    for i in range(0,10):
        tic = time.perf_counter()
        nlist=call_nlist(x,y,z,nlist,1032)
        toc = time.perf_counter()
        print(f"Time = {toc - tic:0.4f} seconds")
    #Call latte to get the density matrix
    #ham, dm = get_latte_dm(box,symbols,types,coords)




