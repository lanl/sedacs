"""Charges
Some functions to compute charges

So far: Initital "get_charges" and "collect_charges"
for nonorthogonal TB"
"""

import numpy as np
from sedacs.message import * 

__all__ = ["get_charges","collect_charges"]

##Get charges for every atom
# @brief This function will return the point Milliken populations 
# for every atom in the system.
# @todo Move this to the proxy code
# @param rho Density matrix 
# @param znuc Number of valence electrons for every atom type.
# @param types Type index for every atom in the system.
# @param part Atom indices within the part (only cores). 
# @param hindex Begining and ending of every index for atomic orbital block.
# Atom i will have its orbitals indexed in between (hindex[i],hindex[i+1])
# @param over Overlap matrix 
# @param verb Verbosity level
#
def get_charges(rho,znuc,types,part,hindex,over=None,verb=False):
    ncores = len(part)
    if(verb):
        status_at("get_charges","Getting charges from density matrix")

    fullDiag = np.diag(rho)
    charges = np.zeros((ncores))

    nel = sum(znuc[types[:]])
    trRho = 0.0
    if(over is None):
        for i in range(ncores): 
            for ii in range(hindex[i],hindex[i+1]):
                charges[i] = charges[i] + fullDiag[ii]
            charges[i] = 2.0*charges[i] - znuc[types[i]]
    else: #S x D
        aux = np.dot(over,rho)
        fullDiag = np.diag(aux)
        for i in range(ncores):
            charges[i] = 0.0
            for ii in range(hindex[i],hindex[i+1]):
                charges[i] = charges[i] + fullDiag[ii]
                trRho = trRho + fullDiag[ii]
            charges[i] = 2.0*charges[i] - znuc[types[i]]
    if(verb):
        msg = "Total Charge for part= " + str(sum(charges))
        status_at("get_charges",msg)

    return charges

## Collect charges at the current rank
# @param chargesOnRank Full charge vetor on every rank. 
# @param charges Charges to be collected.
# @param part list of atom indices in the part.
# @param nats Number of total atoms on the rank
#
def collect_charges(chargesOnRank,charges,part,nats,verb=False):
    if(verb):
        status_at("collect_charges","Collecting charges")

    if(chargesOnRank is None):
        chargesOnRank = np.zeros((nats))

    for i in range(len(part)):
        chargesOnRank[part[i]] = charges[i]

    return chargesOnRank
    

