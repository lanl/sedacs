"""
entropy.py
====================================
Electronic entropy. This module will handle functions 
related to the computation of electronic entropy or Fermi 
Dirac distribution.

"""

import numpy as np


__all__ = [
    "get_entropy",
    "fermi_dirac",
]


def fermi_dirac(mu, energy, temp, kB=8.61739e-5):
    '''
        Get Fermi probability distributions (values are between 0 and 1)
    '''
    fermi = np.where((energy - mu)/(kB*temp) < 100, 1/(1 + np.exp((energy - mu)/(kB*temp))), 0.0)

    return fermi


def get_entropy(mu, evals, etemp, dvals, kB=8.61739e-5, verb=False):
    
    if(verb):
        print('\nCalculating electronic entropy ...,')

    fermi = fermi_dirac(mu, evals, etemp, kB=kB)
    epsilon = 1e-9
    mask = np.logical_and( abs(fermi) > epsilon, abs(1 - fermi) > epsilon)
    dvals = dvals[mask]
    fermi = fermi[mask]
    entropy = np.sum( 2.0*kB*etemp * dvals * (fermi*np.log(fermi) + (1.0 - fermi)*np.log(1.0-fermi)) )

    return entropy
