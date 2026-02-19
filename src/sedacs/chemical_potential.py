"""
chemical_potential.py
====================================
Chemical potential. This module will handle functions 
related to the computation of chemical potential or Fermi 
Dirac distribution.

"""

import numpy as np
from numba import njit


__all__ = [
    "get_mu",
    "fermi_dirac",
]


@njit(cache=True, fastmath=True)
def fermi_dirac(mu, evals, etemp, kB=8.61739e-5):
    """
    Vector Fermi-Dirac occupations for energies `evals` at chemical potential `mu`.
    Uses a numerically-stable exp clamp.
    """
    beta = 1.0 / (kB * etemp)
    n = evals.shape[0]
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        x = (evals[i] - mu) * beta

        # Clamp to avoid overflow in exp for large |x|
        if x > 60.0:
            out[i] = 0.0
        elif x < -60.0:
            out[i] = 1.0
        else:
            out[i] = 1.0 / (1.0 + np.exp(x))
    return out


@njit(cache=True, fastmath=True)
def get_mu_numba(mu0, evals, etemp, nocc, dvals, kB=8.61739e-5, verb=False):
    """
    Numba version of your get_mu:
      - Newton-Raphson iterations first
      - if diverges / doesn't converge => fallback to scan+bisect style you had

    Parameters
    ----------
    mu0 : float
    evals : 1D float64 array
    etemp : float
    nocc : float
    dvals : 1D float64 array (must be provided for numba; pass np.ones_like(evals) if needed)
    kB : float
    verb : bool (printing is supported in numba, but can slow things down)
    """
    if verb:
        print("\nCalculating mu ...,")

    a = 1.0
    nmax = 30
    tol = 1.0e-10

    mu = mu0
    norbs = evals.shape[0]
    notConverged = False

    # ---------- Newton-Raphson ----------
    occErr = 0.0
    occ = 0.0

    for it in range(nmax + 1):
        fermi = fermi_dirac(mu, evals, etemp, kB)
        occ = np.sum(fermi * dvals)
        occErr = abs(occ - nocc)

        if occErr < tol:
            break

        # d occ / d mu = sum_i [ (beta * f_i * (1-f_i)) * dvals_i ]
        beta = 1.0 / (kB * etemp)
        occ_prime = 0.0
        for j in range(norbs):
            fj = fermi[j]
            occ_prime += (beta * fj * (1.0 - fj)) * dvals[j]

        mu = mu + a * (nocc - occ) / (occ_prime + 1.0e-3)

        if abs(mu) > 1.0e10:
            print("WARNING: Newton-Raphson did not converge (will try bisection) Occupation error = ", occErr)
            notConverged = True
            break

        if verb:
            print("N-R iteration (i,mu,occ,occErr)", it, mu, occ, occErr)

        if it == nmax:
            print("WARNING: Newton-Raphson did not converge (will try bisection) Occupation error = ", occErr)
            notConverged = True

    # ---------- Fallback: your scan + step-halving ----------
    if notConverged:
        muMin = np.min(evals)
        muMax = np.max(evals)

        mu = muMin
        step = abs(muMax - muMin)

        # ft = occ(mu) - nocc
        fermi = fermi_dirac(mu, evals, etemp, kB)
        ft1 = np.sum(fermi * dvals) - nocc

        ft2 = ft1

        for it in range(1_000_001):
            if it == 1_000_000:
                print("Bisection method not converging ...")
                # keep behavior close to original; return best guess
                break

            if (mu > muMax + 1.0) or (mu < muMin - 1.0):
                print("Bisection method is diverging")
                print("muMin=", muMin, "muMax=", muMax)
                break

            if abs(ft1) < tol:
                occErr = abs(ft1)
                break

            mu = mu + step

            fermi = fermi_dirac(mu, evals, etemp, kB)
            occ = np.sum(fermi * dvals)
            ft2 = occ - nocc

            prod = ft2 * ft1
            if prod < 0.0:
                mu = mu - step
                step = step * 0.5
            else:
                ft1 = ft2

            if verb:
                print("Bisection iteration (i,mu,occ,occErr);", it, mu, occ, ft2)

        occErr = abs(ft2)

    print("Final mu, error:", mu, occErr)
    return mu


# Convenience wrapper if you want dvals=None at the python level (still numba-compiled core):
def get_mu(mu0, evals, etemp, nocc, dvals=None, kB=8.61739e-5, verb=False):
    evals = np.asarray(evals, dtype=np.float64)
    if dvals is None:
        dvals = np.ones(evals.shape[0], dtype=np.float64)
    else:
        dvals = np.asarray(dvals, dtype=np.float64)
    return get_mu_numba(mu0, evals, float(etemp), float(nocc), dvals, float(kB), bool(verb))


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

    




