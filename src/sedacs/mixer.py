"""mixer 
Some mixing schemes functions to accelerate SCF convergence.

So far: linear_mix, diis_mix  
"""

from sedacs.message import *
from sedacs.periodic_table import PeriodicTable
import numpy as np

from sedacs.types import ArrayLike

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

def diis_mix(charges: ArrayLike,
             chargesOld: ArrayLike,
             chargesIn: ArrayLike,
             chargesOut: ArrayLike,
             iteration: int,
             verb: bool = False,
             mStoring: int = 5,
             mixCoeff: float = 0.2) -> tuple[float, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:

    """

    Performs DIIS mixing scheme

    Parameters
    ----------  
    charges : ArrayLike (Natoms)
        The charges of the system.
    chargesOld : ArrayLike (Natoms)
        The old charges of the system.
    chargesIn : ArrayLike (Natoms, mStoring)
        The input charges of the system.
    chargesOut : ArrayLike (Natoms, mStoring)
        The output charges of the system.
    iteration : int
        The iteration number.
    verb : bool
        Verbosity level.
    mStoring : int
        The number of historical charges to store.
    mixCoeff : float
        The mixing coefficient.

    Returns
    -------
    scfError : float
        The SCF error.
    charges : ArrayLike (Natoms)
        The modified charges of the system.
    chargesOld : ArrayLike (Natoms)
        The old charges of the system.
    chargesIn : ArrayLike (Natoms, mStoring)
        The input charges of the system.
    chargesOut : ArrayLike (Natoms, mStoring)
        The output charges of the system.   
    """

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

        # Linear mixing.
        charges = (1.0 - mixCoeff)*chargesOld + mixCoeff*charges

        # Compute the SCF error.
        scfError = np.linalg.norm(charges)

        if(verb):
            print("SCF error =", scfError)

        # Update the old charges.
        chargesOld[:] = charges[:]

    else:

        # Initialize the auxiliary charges.
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


        # Loop once history is full.
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

        scfError = np.linalg.norm(charges - chargesOld) / np.sqrt(len(charges))

        if(verb):
            print("SCF error =", scfError)

        charges=chargesAux

        chargesOld=chargesAux

    return scfError, charges, chargesOld, chargesIn, chargesOut


def linear_mix(mixCoeff: float,
               charges: ArrayLike,
               chargesOld: ArrayLike,
               iteration: int) -> tuple[float, ArrayLike, ArrayLike]:

    """
    Performs linear mixing scheme

    Parameters
    ----------
    mixCoeff : float
        The mixing coefficient.
    charges : ArrayLike (Natoms)
        The charges of the system.
    chargesOld : ArrayLike (Natoms)
        The old charges of the system.
    iteration : int
        The iteration number.
    verb : bool
        Verbosity level.

    Returns
    -------
    scfError : float
        The SCF error.
    charges : ArrayLike (Natoms)
        The modified charges of the system.
    chargesOld : ArrayLike (Natoms)
        The old charges of the system.
    """

    # Linear mixing scheme.
    charges = mixCoeff*charges + (1-mixCoeff)*chargesOld

    # Compute the SCF error.
    scfError = np.linalg.norm(charges -chargesOld) / np.sqrt(len(charges))

    # Update the old charges.
    chargesOld = charges

    return scfError,charges,chargesOld

