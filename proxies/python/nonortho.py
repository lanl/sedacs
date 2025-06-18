import numpy as np
import scipy.linalg as sp

## Inverse overlap factorization
# @brief Constructs inverse overlap factors given the overlap matrix
# @param over Overlap matrix  
# @param method If a particular needs to be used
# @param accel If an accelerater (hardwar/library/programing model) is used.
# @verb Verbosity switch
##
def get_xmat(over,method="Diag",accel="No",verb=False):

    if(verb):
        print("In get_xmat ...")

    hdim = len(over[0,:])
    if(method == "Diag" and accel == "No"):
        e,v = sp.eigh(over)
        s = 1./np.sqrt(e)
        zmat = np.zeros((hdim,hdim))
        for i in range(hdim):
            zmat[i, :] = s[i] * v[:, i]
        zmat = np.matmul(v, zmat)
    elif method == "Cholesky":
        pass
    else:
        print("ERROR: Method not implemented")
        exit(0)

    if verb:
        print("\nZmat Matrix")
        print(zmat)

    return zmat

