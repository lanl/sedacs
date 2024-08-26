#!/usr/bin/env python3

import numpy as np
from sedacs.system import *
from sedacs.file_io import *
from sedacs.periodic_table import PeriodicTable
import ctypes as ct
import os
import scipy.linalg as sp
from gpmd import *
from sedacs.mpi import collect_and_sum_matrices

MPI = None
try:
    from mpi4py import MPI

    is_mpi_available = True
except ImportError:
    is_mpi_available = False


###############################
## Main starts here
###############################

if is_mpi_available:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numranks = comm.Get_size()
else:
    comm = None
    rank = 0
    numranks = 1

#Folders with different set of params
folderName = "run"+str(rank)
my_path = os.getcwd()
new_path = my_path+"/"+folderName
os.chdir(new_path)
verb = 0

#Load the coordinates and atom types
latticeVectors,symbols,atomTypes,coords0 = read_coords_file("H2O.pdb",lib="None",verb=True)

field = np.zeros((3))

#Get forces at coords0 with the params from folder "rank"
err,charges_out,forces0,dipole_out, energyp = gpmd(latticeVectors,symbols,atomTypes,coords0,field,verb)

