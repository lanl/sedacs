#!/usr/bin/env python3

import numpy as np
#import pysqm as ps 
from coordinates import *
from proxy_a import *

coords = get_random_coordinates(4)
nats = len(coords[:,1])
symbols = []
symbols = ["H"] * nats
write_xyz_coordinates(coords,symbols)

H = get_hamiltonian(coords)
N = len(H[:,1])
Nocc = 2
D = get_densityMatrix(H,N,Nocc)

print("Hamiltonian = ",H)
print("Density Matrix = ",D)



#Give the coordinates to the engine
#Get the DM, evals, core contribution to each eigenpair, back

#First version
#evals, evects, nel = ps.run_pysqm(coordinates,symbols) 
#chargesOfTheCoreOnly = ps.run_pysqm(coordinatesOfThisCorePlusHalo,symbolsOfThisCorePlusHalo,numberOfAtomsInThisCore)

# 
#cores = [
#    [True False True]:[True, True, False]
#    [False, True]
#    ]

#evals, dvals, nel, dm = run_pysqm(coordinates,symbols,maskCore[i],ncores,nhalos)







