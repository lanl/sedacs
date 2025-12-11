#!/usr/bin/env python3

import numpy as np
from sedacs.system import *
from sedacs.periodic_table import PeriodicTable
import ctypes as ct
import os
import copy

def sedacs_nlistbox(coords,latticeVectors,nx,ny,nz,rank,numranks,verb):
 
    # Import the shared library
    sedacspartLibFileName = os.path.dirname(__file__) + "/sedacs_part_lib.so"

    sedacspartLib = ct.CDLL(sedacspartLibFileName)
    f = sedacspartLib.sedacs_nlistbox

    nats = len(coords[:,1])

    #Vectorizing 2D arrays for C-Fortran interoperability
    coordsFlat_in = np.zeros(3*nats) #Vectorized coordinates
    boxOfIFlat_out = np.zeros(nats, dtype=np.int32) 

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

    numparts_out = np.zeros(1, dtype=np.int32) 

    #Specify arguments as a pointers to pass to Fortran
    f.argtypes=[ct.c_int,ct.POINTER(ct.c_double),ct.POINTER(ct.c_int),\
            ct.POINTER(ct.c_double),ct.c_int,ct.c_int,ct.c_int,ct.POINTER(ct.c_int),ct.c_int,ct.c_int,ct.c_int]

    #Inputs
    coords_ptr = coordsFlat_in.ctypes.data_as(ct.POINTER(ct.c_double))
    latticeVectors_ptr = latticeVectorsFlat.ctypes.data_as(ct.POINTER(ct.c_double))

    #Outputs
    boxOfI_ptr = boxOfIFlat_out.ctypes.data_as(ct.POINTER(ct.c_int))
    numparts_ptr = numparts_out.ctypes.data_as(ct.POINTER(ct.c_int))

    #Call to the fortran funtion
    err = f(ct.c_int(nats),coords_ptr,boxOfI_ptr,latticeVectors_ptr,\
            ct.c_int(nx),ct.c_int(ny),ct.c_int(nz),numparts_ptr,\
            ct.c_int(verb),ct.c_int(rank),ct.c_int(numranks))

    boxOfI_out = np.zeros(nats, dtype=np.int32)
    boxOfI_out[:] = boxOfIFlat_out[:] - 1
    
    return err, boxOfI_out, int(numparts_out[0]) 

def sedacs_part(whichParts_guess,graph,degs,nparts,verb):
 
    # Import the shared library
    sedacspartLibFileName = os.path.dirname(__file__) + "/sedacs_part_lib.so"

    sedacspartLib = ct.CDLL(sedacspartLibFileName)
    f = sedacspartLib.sedacs_part
    # Shift the indices to Fortran convention
    whichParts_guess = whichParts_guess + 1
    # Remove the first column (graph degree)
    graph = copy.deepcopy(graph[:, 1:])
    nnodes = len(graph[:, 0])
    maxDegs = len(graph[0, :])
    # Shift the indices to Fortran convention
    for i in range(nnodes):
        graph[i, 0:degs[i]] += 1

    #Vectorizing 2D arrays for C-Fortran interoperability
    whichParts_guessFlat_inout = np.zeros(nnodes, dtype=np.int32)
    graphFlat_in = np.zeros(nnodes*maxDegs, dtype=np.int32) #Vectorized graph
    degsFlat_in = np.zeros(nnodes, dtype=np.int32)
 
    whichParts_guessFlat_inout[:] = whichParts_guess[:]
    graphFlat_in[:] = graph.flatten()
    degsFlat_in[:] = degs[:]

    #Specify arguments as a pointers to pass to Fortran
    f.argtypes=[ct.c_int,ct.c_int,ct.POINTER(ct.c_int),\
            ct.POINTER(ct.c_int),ct.POINTER(ct.c_int),ct.c_int,ct.c_int]

    #Inputs/Outputs
    whichParts_guess_ptr = whichParts_guessFlat_inout.ctypes.data_as(ct.POINTER(ct.c_int))
    graph_ptr = graphFlat_in.ctypes.data_as(ct.POINTER(ct.c_int))
    degs_ptr = degsFlat_in.ctypes.data_as(ct.POINTER(ct.c_int))

    #Call to the fortran funtion
    err = f(ct.c_int(nnodes),ct.c_int(maxDegs),whichParts_guess_ptr,graph_ptr,degs_ptr,\
            ct.c_int(nparts),ct.c_int(verb))

    whichParts_guess_out = np.zeros(nnodes, dtype=np.int32)
    whichParts_guess_out[:] = whichParts_guessFlat_inout[:] - 1 
    
    return err, whichParts_guess_out 

