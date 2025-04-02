"""Energy and forces
Some functions to compute energy and forces 

So far: Initital "collect_forces"
for nonorthogonal TB"
"""

import numpy as np
from sedacs.message import * 

__all__ = ["collect_energy","collect_forces"]


## Collect energy at the current rank
# @param energyOnRank Energy on every rank. 
# @param energy Energy to be collected.
#
def collect_energy(energyOnRank,energy,verb=False):
    if(verb):
        status_at("collect_energy","Collecting energy")

    if(energyOnRank is None):
        energyOnRank = np.zeros((1))

    energyOnRank[0] += energy 

    return energyOnRank


## Collect forces at the current rank
# @param forcesOnRank Full force matrix on every rank. 
# @param forces Forces to be collected.
# @param part list of atom indices in the part.
# @param nats Number of total atoms on the rank
#
def collect_forces(forcesOnRank,forces,part,nats,verb=False):
    if(verb):
        status_at("collect_forces","Collecting forces")

    if(forcesOnRank is None):
        forcesOnRank = np.zeros((nats, 3))

    for i in range(len(part)):
        forcesOnRank[part[i]] = forces[i]

    return forcesOnRank
    

