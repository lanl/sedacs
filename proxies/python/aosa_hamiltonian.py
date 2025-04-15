"""AOSA and LATTE - prototype hamiltonian elements
Atomic orbital spherical approximation
    - Reads the total number of atoms
    - Constructs a set of random coordinates
    - Constructs a simple Hamiltonian
    - Computes the hamiltonian derivatives
"""

import os
import sys

import numpy as np
import scipy.linalg as sp
import sedacs.driver
#import sedacs.interface_modules
from sedacs.dev.io import src_path

try:
    import ctypes

    # import gpulibInterface as gpu

    gpuLib = True
    arch = "nvda"
    pwd = os.getcwd()

    if arch == "nvda":
        print("loading nvidia...")
        lib = ctypes.CDLL(str((src_path() / "gpu/nvda/libnvda.so").absolute()))
    if arch == "amd":
        lib = ctypes.CDLL(str((src_path() / "gpu/amd/libamd.so").absolute()))

except:
    gpuLib = False


__all__ = [
    "AOSA_Parameters",
    "AOSA_Parameter",
    "get_integral",
    "get_integral_v1",
]

class AOSA_Parameter:
    def __init__(self,symbol,orb,filePath):
        self.symbol = symbol
        self.orbType = orb
        parFile = open(filePath,"r")
        count = 0
        symbFound = False
        orbFound = False
        for line in parFile:
            info = line.split()
            if(len(info) >= 1):
                print(info)
                if(info[0] == "Element="):
                    if(info[1] == symbol):
                        norbs = int(info[3])
                        symbFound = True
                    print(info[1],symbol,symbFound)
                if(symbFound and info[3] == orb):
                    print(info[5])
                    orbFound = True 
                    self.onsite = float(info[5])
                    self.u = float(info[7])
                    self.nl = int(info[9])
                    self.kappas = np.zeros((self.nl))
                    self.ds = np.zeros((self.nl,3))
                    self.gammas = np.zeros((self.nl,4))
                if(symbFound and orbFound and info[0] == "LobeIndex="):
                    self.kappas[count] = float(info[3])
                    self.ds[count,0] = float(info[5]) 
                    self.ds[count,1] = float(info[6])
                    self.ds[count,2] = float(info[7])

                    self.gammas[count,0] = float(info[9])
                    self.gammas[count,1] = float(info[10])
                    self.gammas[count,2] = float(info[11])
                    self.gammas[count,3] = float(info[12])

                    count = count + 1

                    if(count == self.nl):
                        break
        parFile.close()  


def get_integral(coordsI,symbolI,orbI,coordsJ,symbolJ,orbJ):
    
        parI = AOSA_Parameters(symbolI,orbI)
        parJ = AOSA_Parameters(symbolJ,orbJ)

        RIJ = coordsJ - coordsI
        #Expo
        inte = 0.0
        for li in range(parI.nl):
            for lj in range(parJ.nl):
                sn = np.sign(parI.kappas[li])*np.sign(parJ.kappas[lj])
                kappaIJ = sn*(abs(parI.kappas[li]) + abs(parJ.kappas[lj]))/2
                gammaIJ = (parI.gammas[li,0] + parI.gammas[li,0])/2
                dIJ = RIJ + parJ.ds[lj,:] - parI.ds[li,:]
                #dIJ = np.dot(RIJ,parJ.ds[lj,:]) + parJ.ds[lj,:] - parI.ds[li,:]
                inte = inte + kappaIJ*np.exp(gammaIJ*np.linalg.norm(dIJ))
    
        sval = inte
        hval = inte*(parI.onsite + parJ.onsite)/2

        return hval, sval



def get_integral_v1(coordsI,coordsJ,parI,parJ):

        RIJ = coordsJ - coordsI
        #Expo
        inte = 0.0
        for li in range(parI.nl):
            for lj in range(parJ.nl):
                sn = np.sign(parI.kappas[li])*np.sign(parJ.kappas[lj])
                kappaIJ = sn*(abs(parI.kappas[li]) + abs(parJ.kappas[lj]))/2
                gammaIJ = (parI.gammas[li,0] + parJ.gammas[lj,0])/2
                dIJ = RIJ + parJ.ds[lj,:] - parI.ds[li,:]
                inte = inte + kappaIJ*np.exp(gammaIJ*np.linalg.norm(dIJ))

        sval = inte
        hval = inte*(parI.onsite + parJ.onsite)/2

        return hval, sval


