#!/usr/bin/env python3

import numpy as np
import time
import gpulibInterface as gpulib
import os 

# to call cuda/hip
import numpy.ctypeslib as npct
import ctypes 


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
    cwd = os.getcwd()
    libnvda = ctypes.CDLL(str(cwd)+"/nvda/libnvda.so")
    #libnvda.diagonalize.argtypes = ctypes.POINTER(ctypes.c_double),

    # Import module
    #pwd="/vast/home/finkeljo/ctypes/simplest/generalcodes/py-to-cpp/"
    #lib = ctypes.CDLL(str(cwd)+"/nlist.so")
    #lib.nlist.argtypes = ctypes.POINTER(ctypes.c_double),

    myVerb = True
    #Read coordinates from pdb file 
    #box,symbols,types,coords = read_pdb_file("coords_1032.pdb",lib="None",verb=myVerb)
 
    #x = coords[:,0]
    #y = coords[:,1]
    #z = coords[:,2]
    

    ham = np.ones((1032*1032,1))
    dm = np.zeros((1032*1032,1))
 
    for i in range(0,10):

        tic = time.perf_counter()
        dm = gpulib.dmDiag(ham,dm,1032,512,"nvidia",libnvda)
        toc = time.perf_counter()
        print(f"Time from main = {toc - tic:0.4f} seconds")





    #Call latte to get the density matrix
    #ham, dm = get_latte_dm(box,symbols,types,coords)




