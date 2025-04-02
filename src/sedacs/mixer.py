"""mixer 
Some mixing schemes functions to accelerate SCF convergence.

So far: linear_mix, diis_mix  
"""

from sedacs.message import *
from sedacs.periodic_table import PeriodicTable
import numpy as np

# from sdc_out import *
try:
    from mpi4py import MPI

    mpiLib = True
except ImportError as e:
    mpiLib = False
from multiprocessing import Pool

if mpiLib:
    from sedacs.mpi import *
import time

__all__ = [
    "linear_mix",
    "diis_mix",
]

## Do DIIS mixing scheme
# @param 
# @param 
# @param verb Verbosity level
#
def diis_mix(charges,chargesOld,chargesIn,chargesOut,iteration,verb=False):


    mStoring = 5 
    mixCoeff = 0.2 

    if(verb):
        status_at("diis_mix","Performing Pulay mixing scheme ...")

    nats = len(charges)

    if(iteration == 0):
        chargesOld = np.zeros((nats))
        chargesOld[:] = charges[:]
        chargesIn = np.zeros((nats,mStoring)) #dqin
        chargesOut = np.zeros((nats,mStoring)) 

    kStoring=min(iteration,mStoring) 

    if(iteration <= 0):
        charges = (1.0 - mixCoeff)*chargesOld + mixCoeff*charges #Linear mixing
        scfError = np.linalg.norm(charges)
        if(verb):
            print("SCF error =", scfError)
        chargesOld[:] = charges[:]
    else:

        chargesAux = np.zeros(nats) #d
        chargesNewIn = np.zeros(nats) #dnew
        chargesNewOut = np.zeros(nats) #dnewOut

        chargesAux[:] = charges[:]

        coeffMat = np.zeros((kStoring+1,kStoring+1))
        bVect = np.zeros((kStoring+1))

        #Shifting the storing vectors
        if(iteration <= mStoring):
            chargesIn[:,iteration-1] = chargesOld[:]
            chargesOut[:,iteration-1] = chargesAux[:]

        if(iteration >= mStoring + 1):

            for j in range(0,kStoring-1):
                chargesIn[:,j] = chargesIn[:,j+1]
                chargesOut[:,j] = chargesOut[:,j+1]

            chargesIn[:,kStoring-1] = chargesOld[:]
            chargesOut[:,kStoring-1] = chargesAux[:]


        coeffMat[:,:] = 0.0

        for i in range(0,kStoring+1):
            coeffMat[kStoring,i] = -1.0
            coeffMat[i,kStoring] = -1.0
            bVect[i] = 0

        bVect[kStoring] = -1.0
        coeffMat[kStoring, kStoring] = 0.0


        for i in range (0,kStoring):
            for j in range (0,kStoring):
                for k in range(nats):
                    coeffMat[i,j] = coeffMat[i,j] + (chargesOut[k,i]-chargesIn[k,i])*(chargesOut[k,j]-chargesIn[k,j])

        if(verb):
            print("Coeffs")
            print(coeffMat)

        try:
            bVect = np.linalg.solve(coeffMat,bVect)
            bSolved = True
        except:
            print('WARNING: Singular matrix in DIIS. Doing linear mixing')
            bVect[:] = 1.0/(float(len(bVect)))
            bSolved = False

        chargesNewIn[:] = np.zeros((nats))
        chargesNewOut[:] = np.zeros((nats))


        if(bSolved):
            for j in range(kStoring):
                chargesNewIn[:] = chargesNewIn[:] + bVect[j] * chargesIn[:,j]
                chargesNewOut[:] = chargesNewOut[:] + bVect[j] * chargesOut[:,j]
        else:
            for j in range(kStoring):
                chargesNewIn[:] = chargesIn[:,j]
                chargesNewOut[:] = chargesOut[:,j]

        chargesAux = (1.0 - mixCoeff)*chargesNewIn + mixCoeff*chargesNewOut

        scfError = np.linalg.norm(charges -chargesOld)

        if(verb):
            print("SCF error =", scfError)

        charges=chargesAux

        chargesOld=chargesAux

    return scfError, charges, chargesOld, chargesIn, chargesOut

## Do linear mixing 
# @param mixCoeff Mixing coefficient
# @param charges System charges
# @param chargesOld Old system charges
# @return charges Mofified system charges according to mixing scheme
# @param verb Verbosity level
#
def linear_mix(mixCoeff,charges,chargesOld,iteration):
    
    if(iteration == 0):
        chargesOld = charges
        scfError = 1.0
    else:
        charges = mixCoeff*charges + (1-mixCoeff)*chargesOld
        scfError = np.linalg.norm(charges -chargesOld)
        chargesOld = charges

    return scfError,charges,chargesOld

