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
#import sedacs.driver
#import sedacs.interface_modules
#from sedacs.dev.io import src_path

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


def LoadBondIntegralParameters_H(TYPE_PAIR):

    TYPE_a = TYPE_PAIR[0]
    TYPE_b = TYPE_PAIR[1]
    
    fss_sigma = np.zeros((8))
    fsp_sigma = np.zeros((8))
    fps_sigma = np.zeros((8))
    fpp_sigma = np.zeros((8))
    fpp_pi = np.zeros((8))
    
    if TYPE_a == 'H':
        if TYPE_b == 'H':
            fss_sigma[0:8] = [-9.400000,-1.145903,-0.391777,0.000,0.000,0.750,3.500,4.000]
            fsp_sigma[0:8] = [0,0,0,0,0,0,0,0]
            fps_sigma[0:8] = [0,0,0,0,0,0,0,0]
            fpp_sigma[0:8] = [0,0,0,0,0,0,0,0]
            fpp_pi[0:8] = [0,0,0,0,0,0,0,0]
            Es = -6.4835
            Ep = 0.0
            U = 12.054683
        elif TYPE_b == 'C':
            fss_sigma[0:8] = [-9.235812,-1.372683,-0.408433,0.0000,0.0000,1.1000,3.5000,4.000 ]
            fsp_sigma[0:8] = [8.104851 ,-0.936099 ,-0.626219 ,0.000 ,0.000 ,1.100 ,3.500 ,4.000 ]
            fps_sigma[0:8] = [8.104851 ,-0.936099 ,-0.626219 ,0.000 ,0.000 ,1.100 ,3.500 ,4.000 ]
            fpp_sigma[0:8] = [0,0,0,0,0,0,0,0]
            fpp_pi[0:8] = [0,0,0,0,0,0,0,0]
            Es = 0; Ep = 0; U = 0
        elif TYPE_b == 'O':
            fss_sigma[0:8] = [-12.189103 ,-1.800097 ,-0.325933 ,0.000 ,0.000 ,1.000 ,3.500 ,4.000 ]
            fsp_sigma[0:8] = [9.518733 ,-1.333235 ,-0.39371 ,0.000 ,0.000 ,1.000 ,3.500 ,4.000 ]
            fps_sigma[0:8] = [9.518733 ,-1.333235 ,-0.39371 ,0.000 ,0.000 ,1.000 ,3.500 ,4.000 ]
            fpp_sigma[0:8] = [0,0,0,0,0,0,0,0]
            fpp_pi[0:8] = [0,0,0,0,0,0,0,0]
            Es = 0; Ep = 0; U = 0
        elif TYPE_b == 'N':
            fss_sigma[0:8] = [-12.63103 ,-1.585597 ,-0.250969 ,0.00 ,0.000 ,1.00 ,3.50 ,4.00 ]
            fsp_sigma[0:8] = [9.837852 ,-1.23485 ,-0.324283 ,0.00 ,0.000 ,1.00 ,3.50 ,4.00 ]
            fps_sigma[0:8] = [9.837852 ,-1.23485 ,-0.324283 ,0.00 ,0.000 ,1.00 ,3.50 ,4.00 ]
            fpp_sigma[0:8] = [0,0,0,0,0,0,0,0]
            fpp_pi[0:8] = [0,0,0,0,0,0,0,0]
            Es = 0; Ep = 0; U = 0
    elif TYPE_a == 'C':
        if TYPE_b == 'H':
            fss_sigma[0:8] = [-9.235812 ,-1.372683 ,-0.408433 ,0.00 ,0.00 ,1.10 ,3.50 ,4.00 ]
            fsp_sigma[0:8] = [8.104851 ,-0.936099 ,-0.626219 ,0.00 ,0.00 ,1.10 ,3.50 ,4.00 ]
            fps_sigma[0:8] = [8.104851 ,-0.936099 ,-0.626219 ,0.00 ,0.00 ,1.10 ,3.50 ,4.00 ]
            fpp_sigma[0:8] = [0,0,0,0,0,0,0,0]
            fpp_pi[0:8] = [0,0,0,0,0,0,0,0]
            Es = 0; Ep = 0; U = 0
        elif TYPE_b == 'C':
            fss_sigma[0:8] = [-9.197237 ,-1.60705 ,-0.535057 ,0.00 ,0.00 ,1.40 ,3.50 ,4.00 ]
            fsp_sigma[0:8] = [8.562436 ,-0.980182 ,-0.646929 ,0.00 ,0.00 ,1.40 ,3.50 ,4.00 ]
            fps_sigma[0:8] = [8.562436 ,-0.980182 ,-0.646929 ,0.00 ,0.00 ,1.40 ,3.50 ,4.00 ]
            fpp_sigma[0:8] = [6.614756 ,-0.528591 ,-0.95146 ,0.00 ,0.00 ,1.40 ,3.50 ,4.00 ]
            fpp_pi[0:8] = [-3.678302 ,-1.881668 ,-0.255951 ,0.00 ,0.00 ,1.40 ,3.50 ,4.00 ]
            Es = -13.7199  
            Ep = -5.2541  
            U = 14.240811
        elif TYPE_b == 'O':
            fss_sigma[0:8] = [-13.986685 ,-1.931973 ,-0.432011 ,0.00 ,0.00 ,1.20 ,3.50 ,4.00 ]
            fsp_sigma[0:8] = [10.718738 ,-1.389459 ,-0.182128 ,0.00 ,0.00 ,1.20 ,3.50 ,4.00 ]
            fps_sigma[0:8] =  [14.194791 ,-1.37165 ,-0.248285 ,0.00 ,0.00 ,1.20 ,3.50 ,4.00 ]
            fpp_sigma[0:8] = [8.622023 ,-0.557144 ,-0.938551 ,0.00 ,0.00 ,1.20 ,3.50 ,4.00 ]
            fpp_pi[0:8] = [-5.327397 ,-2.19016 ,-0.089303 ,0.00 ,0.00 ,1.20 ,3.50 ,4.00 ]
            Es = 0; Ep = 0; U = 0;
        elif TYPE_b == 'N':
            fss_sigma[0:8] = [-7.409712 ,-1.940942 ,-0.219762 ,0.00 ,0.00 ,1.50 ,3.50 ,4.00 ]
            fsp_sigma[0:8] = [7.501761 ,-1.211169 ,-0.373905 ,0.00 ,0.00 ,1.50 ,3.50 ,4.00 ]
            fps_sigma[0:8] =  [8.697591 ,-1.26724 ,-0.178484 ,0.00 ,0.00 ,1.50 ,3.50 ,4.00 ]
            fpp_sigma[0:8] = [6.95460 ,-1.188456 ,-0.808043 ,0.00 ,0.00 ,1.50 ,3.50 ,4.00 ]
            fpp_pi[0:8] = [-2.921605 ,-2.203548 ,-0.409424 ,0.00 ,0.00 ,1.50 ,3.50 ,4.00 ]
            Es = 0; Ep = 0; U = 0
    elif TYPE_a == 'O':
        if TYPE_b == 'H':
            fss_sigma[0:8] = [-12.189103 ,-1.800097 ,-0.325933 ,0.00 ,0.00 ,1.00 ,3.50 ,4.00 ]
            fsp_sigma[0:8] = [9.518733 ,-1.333235 ,-0.39371 ,0.00 ,0.00 ,1.00 ,3.50 ,4.00 ]
            fps_sigma[0:8] = [9.518733 ,-1.333235 ,-0.39371 ,0.00 ,0.00 ,1.00 ,3.50 ,4.00 ]
            fpp_sigma[0:8] = [0,0,0,0,0,0,0,0]
            fpp_pi[0:8] = [0,0,0,0,0,0,0,0]
            Es = 0; Ep = 0; U = 0
        elif TYPE_b == 'C':
            fss_sigma[0:8] = [-13.986685 ,-1.931973 ,-0.432011 ,0.00 ,0.00 ,1.20 ,3.50 ,4.00 ]
            fsp_sigma[0:8] = [14.194791 ,-1.37165 ,-0.248285 ,0.00 ,0.00 ,1.20 ,3.50 ,4.00 ]
            fps_sigma[0:8] = [10.718738 ,-1.389459 ,-0.182128 ,0.00 ,0.00 ,1.20 ,3.50 ,4.00 ]
            fpp_sigma[0:8] = [8.622023 ,-0.557144 ,-0.938551 ,0.00 ,0.00 ,1.20 ,3.50 ,4.00 ]
            fpp_pi[0:8] = [-5.327397 ,-2.19016 ,-0.089303 ,0.00 ,0.00 ,1.20 ,3.50 ,4.00 ]
            Es = 0; Ep = 0; U = 0
        elif TYPE_b == 'O':
            fss_sigma[0:8] = [-14.387756 ,-2.244278 ,-1.645605 ,0.0 ,0.0 ,1.2 ,3.5 ,4.0 ]
            fsp_sigma[0:8] = [13.699127 ,-1.602358 ,-0.114474 ,0.0 ,0.0 ,1.2 ,3.5 ,4.0 ]
            fps_sigma[0:8] = [13.699127 ,-1.602358 ,-0.114474 ,0.0 ,0.0 ,1.2 ,3.5 ,4.0 ]
            fpp_sigma[0:8] = [9.235469 ,-1.131474 ,-0.924535 ,0.0 ,0.0 ,1.2 ,3.5 ,4.0 ]
            fpp_pi[0:8] = [ -4.526526 ,-2.487174 ,-0.201464 ,0.0 ,0.0 ,1.2 ,3.5 ,4.0 ]
            Es = -23.9377
            Ep = -9.0035  
            U = 11.8761410
    elif TYPE_b == 'N':
            fss_sigma[0:8] = [-9.360078 ,-1.293118 ,-0.379415 ,0.00 ,0.00 ,1.20 ,3.50 ,4.00 ]
            fsp_sigma[0:8] = [10.723048 ,-0.454312 ,-0.916563 ,0.00 ,0.00 ,1.20 ,3.50 ,4.00 ]
            fps_sigma[0:8] = [10.309052 ,-0.981652 ,-0.828497 ,0.00 ,0.00 ,1.20 ,3.50 ,4.00 ]
            fpp_sigma[0:8] = [9.259131 ,-0.734112 ,-1.023762 ,0.00 ,0.00 ,1.20 ,3.50 ,4.00 ]
            fpp_pi[0:8] = [-4.532623 ,-1.999631 ,-0.286275 ,0.00 ,0.00 ,1.20 ,3.50 ,4.00 ]
            Es = 0; Ep = 0; U = 0
    elif TYPE_a == 'N':
        if TYPE_b == 'H':
            fss_sigma[0:8] = [-12.63103 ,-1.585597 ,-0.250969 ,0.00 ,0.00 ,1.00 ,3.50 ,4.00 ]
            fsp_sigma[0:8] = [9.837852 ,-1.23485 ,-0.324283 ,0.00 ,0.00 ,1.00 ,3.50 ,4.00 ]
            fps_sigma[0:8] = [9.837852 ,-1.23485 ,-0.324283 ,0.00 ,0.00 ,1.00 ,3.50 ,4.00 ]
            fpp_sigma[0:8] = [0,0,0,0,0,0,0,0]
            fpp_pi[0:8] = [0,0,0,0,0,0,0,0]
            Es = 0; Ep = 0; U = 0
        elif TYPE_b == 'C':
            fss_sigma[0:8] = [-7.409712 ,-1.940942 ,-0.219762 ,0.00 ,0.00 ,1.50 ,3.50 ,4.00 ]
            fsp_sigma[0:8] = [8.697591 ,-1.26724 ,-0.178484 ,0.00 ,0.00 ,1.50 ,3.50 ,4.00 ]
            fps_sigma[0:8] = [7.501761 ,-1.211169 ,-0.373905 ,0.00 ,0.00 ,1.50 ,3.50 ,4.00 ]
            fpp_sigma[0:8] = [6.95460 ,-1.188456 ,-0.808043 ,0.00 ,0.00 ,1.50 ,3.50 ,4.00 ]
            fpp_pi[0:8] = [-2.921605 ,-2.203548 ,-0.409424 ,0.00 ,0.00 ,1.50 ,3.50 ,4.00 ]
            Es = 0; Ep = 0; U = 0
        elif TYPE_b == 'O':
            fss_sigma[0:8] = [-9.360078 ,-1.293118 ,-0.379415 ,0.0 ,0.0 ,1.2 ,3.5 ,4.0 ]
            fsp_sigma[0:8] = [10.309052 ,-0.981652 ,-0.828497 ,0.0 ,0.0 ,1.2 ,3.5 ,4.0 ]
            fps_sigma[0:8] = [10.723048 ,-0.454312 ,-0.916563 ,0.0 ,0.0 ,1.2 ,3.5 ,4.0 ]
            fpp_sigma[0:8] = [9.259131 ,-0.734112 ,-1.023762 ,0.0 ,0.0 ,1.2 ,3.5 ,4.0 ]
            fpp_pi[0:8] = [-4.532623 ,-1.999631 ,-0.286275 ,0.0 ,0.0 ,1.2 ,3.5 ,4.0 ]
            Es = 0; Ep = 0; U = 0
        elif TYPE_b == 'N':
            fss_sigma[0:8] = [-7.165811 ,-2.348869 ,-0.541905 ,0.0 ,0.0 ,1.5 ,3.5 ,4.0 ]
            fsp_sigma[0:8] = [8.212268 ,-1.499123 ,-0.52644 ,0.0 ,0.0 ,1.5 ,3.5 ,4.0 ]
            fps_sigma[0:8] = [8.212268 ,-1.499123 ,-0.52644 ,0.0 ,0.0 ,1.5 ,3.5 ,4.0 ]
            fpp_sigma[0:8] = [7.102331 ,-1.252366 ,-0.552533 ,0.0 ,0.0 ,1.5 ,3.5 ,4.0 ]
            fpp_pi[0:8] = [-2.828938 ,-2.376886 ,-0.560898 ,0.0 ,0.0 ,1.5 ,3.5 ,4.0 ]
            Es = -18.5565
            Ep = -7.0625
            U = 17.3729
    else:
        Es = 0; Ep = 0; U = 0
        fss_sigma[0:8] = [0,0,0,0,0,0,0,0]
        fss_sigma[0:8] = [0,0,0,0,0,0,0,0]
        fsp_sigma[0:8] = [0,0,0,0,0,0,0,0]
        fps_sigma[0:8] = [0,0,0,0,0,0,0,0]
        fpp_sigma[0:8] = [0,0,0,0,0,0,0,0]
        fpp_pi[0:8] = [0,0,0,0,0,0,0,0]
    
    #Maybe better to pre-calculate?
    fss_sigma = ScaleTail(fss_sigma)
    fsp_sigma = ScaleTail(fsp_sigma)
    fps_sigma = ScaleTail(fps_sigma)
    fpp_sigma = ScaleTail(fpp_sigma)
    fpp_pi = ScaleTail(fpp_pi)
    
    return fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,Es,Ep,U


def ScaleTail(A):
    if abs(A[0]) < 1e-12:
        A[8:13] = 0;
    else:
        R1 = A[6]
        RCUT = A[7]
        R1SQ = R1*R1
        RMOD = R1 - A[5]
        POLYNOM = RMOD*(A[1] + RMOD*(A[2] + RMOD*(A[3] + A[4]*RMOD)))
        SCL_R1 = exp(POLYNOM)
        DELTA = RCUT - R1
        # Now we're using a 6th order polynomial: fitted to value, first,
        # and second derivatives at R1 and R_cut
        A[8] = SCL_R1
        RMOD = R1 - A[5]
        DPOLY = A[1] + 2*A[2]*RMOD + 3*A[3]*RMOD*RMOD + 4*A[4]*RMOD*RMOD*RMOD
        A[9] = DPOLY*SCL_R1
        DDPOLY = 2*A[2] + 6*A[3]*RMOD + 12*A[4]*RMOD*RMOD
        A[10] = (DPOLY*DPOLY + DDPOLY)*SCL_R1/2
        DELTA2 = DELTA*DELTA
        DELTA3 = DELTA2*DELTA
        DELTA4 = DELTA3*DELTA
        A[11] = (-1/DELTA3)*(3*A[10]*DELTA2 + 6*A[9]*DELTA + 10*A[8])
        A[12] = (1/DELTA4)*(3*A[10]*DELTA2 + 8*A[9]*DELTA + 15*A[8])
        A[13] = (-1/(10*DELTA3))*(6*A[12]*DELTA2 + 3*A[11]*DELTA + A[10])

