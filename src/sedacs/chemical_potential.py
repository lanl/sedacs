import numpy as np
import sys 
import scipy.linalg as sp

kb = 8.61739e-5 # eV/K, kB = 6.33366256e-6 Ry/K, kB = 3.166811429e-6 Ha/K, #kB = 3.166811429e-6 #Ha/K

def fermi_dirac(mu, energy, T):
    '''
    get fermi occupations
    '''
    return 1/(1 + np.exp((energy - mu)/(kb*T)))

def d_fermi_dirac_d_mu(mu, energy, T):
    '''
    derivative of fermi_diract occs
    '''

    expo1 = (np.exp((energy - mu)/(kb*T)))
    #f = 1/(1 + np.exp((energy - mu)/(kb*T)))
    #print('DIFFERENCE: ', (1/kb*T)*(f*(1-f) -  )
    #return (1/(kb*T))*(f*(1-f))
    return (1/(kb*T))  * expo1 / ((1 + expo1))**2
    #return (1/(kb*T))  * expo1 / ((1 + expo1))**2

def get_mu(mu0, dVals, eVals, T, Nocc):
    print('\nCalculating mu:')
    #print(Nocc, dVals.shape, eVals.shape)
    '''
    x_n+1 = x_n  - a*f(x_n)/f`(x_n)
    '''
    a = 1.0

    N_newt_its = 30
    print('#, mu, g:')
    #print(eVals)
    for I in range(N_newt_its):

        f = fermi_dirac(mu0, eVals, T)
        df_dmu = d_fermi_dirac_d_mu(mu0, eVals, T)
        #print('!!!',len(f),  len(dVals), len(eVals))
        g = np.sum([f[i]*dVals[i] for i in range(len(f))]) - Nocc
        if I%4 == 0:
            print('    ', I, mu0, g)
        if abs(g) < 1e-10:
            break

        g_prime = np.sum([df_dmu[i]*dVals[i] for i in range(len(df_dmu))])
        #print('g_prime:', g_prime)
        mu0 = mu0 - a*g/g_prime
    print('Final mu, g:', I, mu0, g)
        
    if I == N_newt_its-1:
        print('WARNING: Newton-Raphson did not converge: abs(g) = ', abs(g))


    return mu0
