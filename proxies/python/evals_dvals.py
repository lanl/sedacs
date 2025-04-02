"""density_matrix 
Computes the Density matrix from a given Hamiltonian
This code is only used to guide implemetations and understand which are the 
basic elements needed to interface with the sedacs driver.
"""

import numpy as np
import scipy.linalg as sp
from nonortho import get_xmat


__all__ = [
    "get_evals_dvals_proxy",
]


## Computes the Evals and Dvals from a given Hamiltonian.
# @author Anders Niklasson
# @brief This will create a Density matrix \f$ \rho \f$
# \f[ \rho  =  \sum^{nocc} v_k v_k^T \f]
# where \f$ v_k \f$ are the eigenvectors of the matrix \f$ H \f$
#
# @param ham Hamiltonian matrix
# @param nocc Number of occupied orbitals
# @param core_size Number of atoms in the cores.
# @param method Type of algorithm used to compute DM
# @param accel Type of accelerator/special device used to compute DM. Default is No and
# will only use numpy.
# @param mu Chemical potential. If set to none, the calculation will use nocc
# @param etemp Electronic temperature
# @param overlap Overlap matrix
# @param verb Verbosity. If True is passed, information is printed.
#
# @return rho Density matrix
#
def get_evals_dvals_proxy(ham, nocc, norbsInCore=None, method="Diag", accel="No", mu=None, etemp=0.0, overlap=None, full_data=False, verb=False, lib=None):
    """Calcualtion of the evals and dvals from H"""
    if verb:
        print("Computing Evals and Dvals")

    norbs = len(ham[:, 0])
    ham_orth = np.zeros((norbs, norbs))
    if overlap is not None:
        # Get the inverse overlap factor
        zmat = get_xmat(overlap, method="Diag", accel="No", verb=False)

        # Orthogonalize Hamiltonian
        ham_orth = np.matmul(np.matmul(np.transpose(zmat), ham), zmat)
    else:
        ham_orth[:, :] = ham[:, :]

    dvals = np.zeros((norbs))
    if(method == "Diag"):
        evals, evects = sp.eigh(ham_orth)
         
        if verb:
            print("Eigenvalues of H:", evals)

        if (overlap is not None):
            #overTimesEvects = np.dot(overlap,evects)
            overTimesEvects = evects
        else:
            overTimesEvects = evects
        for i in range(norbs):
            dvals[i] = np.inner(evects[:norbsInCore,i],overTimesEvects[:norbsInCore,i])
    else:
        if(norbsInCore is not None):
            dvals[:] = norbsInCore/norbs
        else:
            dvals[:] = 1.0

        evals = np.zeros(norbs)
        evals = np.diag(ham_orth) #We estimate the eigenvaluse from the Girshgorin centers


    return evals, dvals

