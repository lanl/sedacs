"""Chemical potential. This module will handle functions 
related to the computation of chemical potential or Fermi 
"""

import os
import sys

import numpy as np
import scipy.linalg as sp
import sedacs.driver
from sedacs.dev.io import src_path
from hamiltonian_elements import *
from sedacs.file_io import read_coords_file, write_xyz_coordinates
from dnnprt import *
from proxy_global import *
from hamiltonian_random import get_random_hamiltonian
from hamiltonian_random import get_random_coordinates
from hamiltonian_random import RandomNumberGenerator


__all__ = [
    "get_mu",
    "fermi_dirac",
]


## Fermi-Dirac function
# @brief Get the Fermi-Dirac distribution probabilities given a set 
# of energy values
# @param mu Chemical potential 
# @param energy Energy value/s 
# @param etemp Electronic temperature [K]
# @param kB Boltzman constant (default is in eV/K)
#
def fermi_dirac(mu, energy, temp, kB=8.61739e-5):
    '''
        Get Fermi probability distributions (values are between 0 and 1)
    '''
    fermi = 1/(1 + np.exp((energy - mu)/(kB*temp)))

    return fermi


## Comput the chemical potential 
# @brief Get the chemical potential from a set of eigenvalues and their weights 
# coputed from a partial trace over a "subsytem". It first uses a Newton-Raphson (NR)
# scheme. It then applies a bisection method if NR does not converge
# @param mu0 Initial guess of mu. If set to None, it will use (HOMO+LUMO)/2. 
# @param evals Eigenvalues of the system 
# @param etemp Electronicn temperature 
# @param nocc Number of occupied orbitals (This is typically coputed from the total 
# number of electrons)
# @param dvals Weights computed from a partial trace. If set to None, weights are set to 1.0.
# @param kB Boltzman constant (default is in eV/K)
# @param verb Verbosity switch 
# 
def get_mu(mu0, evals, etemp, nocc, dvals=None, kB=8.61739e-5, verb=False):
    
    if(verb):
        print('\nCalculating mu ...,')

    a = 1.0
    nmax = 30
    tol = 1.0E-10

    HOMO = evals[int(nocc)]
    LUMO = evals[int(nocc) + 1]
    mu = 0.5*(LUMO + HOMO)
    norbs = len(evals)
    notConverged = False
    if(dvals is None): 
        dvals = np.ones((norbs))
    for i in range(nmax+1):
        fermi = fermi_dirac(mu, evals, temp)
        occ = np.sum([fermi[i]*dvals[i] for i in range(norbs)])
        occErr = abs(occ - nocc)
        dFermiDmu =  (1/(kB*temp))*fermi*(1.0-fermi)*dvals
        occ_prime = np.sum([dFermiDmu[i]*dvals[i] for i in range(norbs)])
        mu = mu + a*(nocc - occ)/occ_prime
        if abs(occErr) < tol:
            break
        elif(abs(mu) > 1.0E10):
            print('WARNING: Newton-Raphson did not converge (will try bisection) Occupation error = ', occErr)
            notConverged = True
            break
        if verb: 
            print('N-R iteration (i,mu,occ,occErr)', i, mu, occ, occErr)
        if(i == nmax):
            print('WARNING: Newton-Raphson did not converge (will try bisection) Occupation error = ', occErr)
            notConverged = True
    
    if(notConverged):
        muMin = np.min(evals)
        muMax = np.max(evals)
        mu = muMin
        step = abs(muMax-muMin)
        Ft1 = 0.0
        Ft2 = 0.0
        prod = 0.0

        #Sum of the occupations
        fermi = fermi_dirac(mu, evals, temp)
        ft1 = np.sum([fermi[i]*dvals[i] for i in range(norbs)])
        ft1 = ft1 - nocc
    
        for i in range(1000001):
            if(i == 1000000):
                print("Bisection method in gpmdcov_musearch_bisec not converging ...")
                exit(0)
            if(mu > muMax + 1.0 or mu < muMin - 1.0):
                print("Bisection method is diverging")
                print("muMin=",muMin,"muMax=",muMax)
                print(evals)
                exit(0)
        
            if(abs(ft1) < tol): #tolerance control
                occErr = ft2
                break
            mu = mu + step

            ft2 = 0.0

            #New sum of the occupations
            fermi = fermi_dirac(mu, evals, temp)
            ft2 = np.sum([fermi[i]*dvals[i] for i in range(norbs)])

            ft2 = ft2 - nocc

            #Product to see the change in sign.
            prod = ft2*ft1
            if(prod < 0):
                mu = mu - step
                step = step / 2.0 #If the root is inside we shorten the step.
            else:
                ft1 = ft2  #If not, Ef moves forward.
            if verb: 
                print('Bisection iteration (i,mu,occ,occErr);', i, mu, occ, ft2)
        
    print('Final mu, error:', mu, occErr)
        
    return mu

## Estimates mu from a matrix using the Girshgorin centers
# @brief It will use the diagonal elements as an approximation 
# for eigenvalues.
# @param ham Hamiltonian matrix 
# @param etemp Electroninc temperature 
# @param nocc Number of occupied states
# @param kB Boltzman constante (default is in units of eV/K)
# @param verb Vorbosity switch 
#
def estimate_mu(ham,etemp,nocc,kB=8.61739e-5,verb=False):
    diag = np.sort(np.diagonal(ham))
    if(verb):
        print("Estimating the chemical potential from diagonal elements ... \n")
    mu0 = 0.5*(np.max(diag) + np.min(diag))
    print("diag",diag)
    print("Mu0",mu0)
    mu = get_mu(mu0,diag,etemp,nocc,kB=kB,dvals=None,verb=True)

    return mu


if __name__ == "__main__":

    n = len(sys.argv)

    if n == 1:
        print("Give the total number of elements. Example:\n")
        print("proxy_a 100\n")
        sys.exit(0)
    else:
        norbs = int(sys.argv[1])

    verb = True

    #Build random coordinates
    coords = get_random_coordinates(norbs)

    #Build random Hamiltonian (Anders' version)
    ham = get_random_hamiltonian(coords)
    
    print("\n Hamiltonan:")
    print(ham)

    nocc = 0.5*norbs
    etemp = 10000

    #Scale the diagonal elements
    scalingFactor = 1.0
    for i in range(norbs):
        ham[i,i] = scalingFactor*ham[i,i]

    #Estimate mu from the diagonal elements of H
    muEst = estimate_mu(ham,etemp,nocc,kB=8.61739e-5)
    print("\n Estimated mu:",muEst)

    #Get eigenvalues and eigenvectors
    evals, evects = sp.eigh(ham)
   
    #Compute exact mu
    mu0 = 0.0
    muReal = get_mu(mu0,evals,etemp,nocc,dvals=None,verb=True)
    print("\n Exact mu:",muEst)

    print("\n Realative error of mu estimation",abs((muReal - muEst)/muReal))
    




