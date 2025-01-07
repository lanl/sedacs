import numpy as np
import sys 
import scipy.linalg as sp
import torch

kb = 8.61739e-5 # eV/K, kB = 6.33366256e-6 Ry/K, kB = 3.166811429e-6 Ha/K, #kB = 3.166811429e-6 #Ha/K

def fermi_dirac(mu0, energy, T):
    '''
    get fermi occupations
    '''
    if len(energy) == 2: # open shell
        return 1/(1 + np.exp((energy - np.expand_dims(mu0, axis=1))/(kb*T)))
    else:
        return 1/(1 + np.exp((energy - mu0)/(kb*T)))

def d_fermi_dirac_d_mu(mu0, energy, T):
    '''
    derivative of fermi_diract occs
    '''
    if len(energy) == 2: # open shell
        expo1 = (np.exp((energy - np.expand_dims(mu0, axis=1))/(kb*T)))
    else: # closed shell
        expo1 = (np.exp((energy - mu0)/(kb*T)))
    #f = 1/(1 + np.exp((energy - mu0)/(kb*T)))
    return (1/(kb*T))  * expo1 / ((1 + expo1))**2

def get_mu(mu0, dVals, eVals, T, Nocc):
    print('\nCalculating mu0:')
    '''
    x_n+1 = x_n  - a*f(x_n)/f`(x_n)
    '''
    
    a = 1.0
    N_newt_its = 30
    print('Iter, mu0, g:')
    for I in range(N_newt_its):
        f = fermi_dirac(mu0, eVals, T)
        df_dmu = d_fermi_dirac_d_mu(mu0, eVals, T)
        if len(dVals) == 2: # open shell
            g = np.sum([f[:,i]*dVals[:,i] for i in range(len(f[0]))], axis=0) - Nocc
            g_prime = np.sum([df_dmu[:,i]*dVals[:,i] for i in range(len(df_dmu[0]))], axis=0)
        else: # closed shell
            g = np.sum([f[i]*dVals[i] for i in range(len(f))]) - Nocc
            g_prime = np.sum([df_dmu[i]*dVals[i] for i in range(len(df_dmu))])
        #if I%4 == 0: print("     {} {:>7.8f} {:>7.8f}".format(I, mu0, g))
        if I%4 == 0: print("     {} {} {}".format(I, mu0, g))
        if (abs(g) < 1e-10).all(): break
        mu0 = mu0 - a*g/g_prime
    print("Final mu0, g: {} {} {}".format(I, mu0, g))
    if I == N_newt_its-1:
        print('WARNING: Newton-Raphson did not converge: abs(g) = ', abs(g))
    return mu0
