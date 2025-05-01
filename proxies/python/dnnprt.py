#!/usr/bin/env python
import numpy as np
import os, sys, argparse, time


def gershgorin(M):
    # find eigenvalue estimates of the matrix M from the Gershgorin circle theorem
    min_e = 0
    max_e = 0

    for i in range(0, np.shape(M)[0]):
        e = M[i, i]  # Gershgorin eigenvalue circle center
        r = 0

        for j in range(0, np.shape(M)[0]):  # compute sum of abs. val of components in row i
            r += np.abs(M[i, j])

        r -= np.abs(e)  # Gershgorin eigenvalue circle radius

        # update min and max eigenvalues as you loop over rows
        if e - r < min_e:
            min_e = e - r
        elif e + r > max_e:
            max_e = e + r

    return (min_e, max_e)


def dual_half(S):
    # Calculation of X^2 as defined in equation (15). Note that precision in numpy is closed under alegbraic operations when of the same data type and inherits the precision of the highest precision operand when not of the same data type.
    S0 = np.half(S)
    S1 = np.single(np.half(S - S0))
    S0S0 = np.single(np.matmul(S0, S0))
    S0S1 = np.matmul(S0, S1)
    X = S0S0 + (S0S1 + np.transpose(S0S1))
    return X

##SP2 method.
# @param h_bml Input Hamiltonian matrix
# @param rho_bml Output density matrix
# @param threshold Threshold for sparse matrix algebra
# @param bndfil Bond
# @param minsp2iter Minimum sp2 iterations
# @param maxsp2iter Maximum SP2 iterations
# @param sp2conv Convergence type
# @param idemtol Idempotency tolerance
# @param verbose A verbosity level
def sp2_basic(ham,occ,thresh=0.0,minsp2iter=5,maxsp2iter=30,sp2conv=1.0E-5,idemtol=1.0E-5,verb=False):

    hdim = len(ham[:,0])

    # Normalize
    emin, emax = gershgorin(ham)
    rho = np.zeros((hdim,hdim))
    rho[:,:] = ham[:,:]
    ident = np.diag(np.ones(hdim))
    rho[:,:] = (emax * ident[:,:] - ham[:,:]) / (emax - emin)

    # X2 <- X
    for i in range(maxsp2iter):

      trx = np.trace(rho)

      #X2 <- X * X
      X2 = np.dot(rho, rho)

      trx2 = np.trace(X2)

      #Trace reduction
      if(verb): print("sp2iter", iter, occ, trx, abs(occ-trx))

      if(trx - occ <= 0.0):

        #X <- 2 * X - X2
        rho = 2*rho - X2

        trx = 2.0 * trx - trx2

      else:

        #X <- X2
        rho[:,:] = X2[:,:]

        trx = trx2


      if(abs(occ-trx) < idemtol and i > minsp2iter):
          break

      if(iter == maxsp2iter):
        print("sp2 purification is not converging: stop!")
        raise 1
      

    #rho = 2.0 * rho

    print("Rho inside",rho)
    return rho 



#### DEEP-NN FORMULATION OF THE RECURSIVE SP2 FERMI-OPERATOR EXPANSION SCHEME
def dnnprt(H0, N, Nocc, H1=None, refi=False):
    """
    Compute density matrix and first order response to perturbation, H1, in the
    Hamiltonian, H0. Implementation mimics mixed precision solver.

    Inputs:
    ------

        H0:         Hamiltonian matrix
        H1:         Perturbation to H0
        N:          Matrix size
        Nocc:       Occupation number
        dm_only:    Compute density matrix only (True/False)
        refi:       Compute fp64 refinement (True/False)

    """
    np.set_printoptions(precision=15)

    dm_only = True
    if H1 is not None:
        dm_only = False

    #### INITIALIZE
    eps = 1e-16  # Small value, but such that +/- eps is finite in single precision
    Csp2 = 4.5  # Convergence criterion as derived by Emanuel
    sgn = 0  # Initial value of sgn
    I = np.eye(N)  # Identity matrix
    maxlayer = 100  # Maximum number of layers
    v_sgn = np.zeros(maxlayer)  # Keeps track of binary in-place learning choices
    idemp_err = np.zeros(maxlayer)  # Local error estimate
    if dm_only == False:
        idemp_err_1 = np.zeros(maxlayer)  # Local error estimate

    #### CHOSE POST-PROCESSING ACTIVATION FUNCTION REFINEMENT OR NOT
    Refinement = True  # Or False

    #### LOAD HAMILTONIAN AS INPUT LAYER
    X0 = H0  # Initial input layer
    if not dm_only:
        X1 = H1

    #### 'EXACT' SOLUTION FOR COMPARISION ONLY
    # e, v = np.linalg.eig(H0 + H1)  # Diagonlize H as a brute force comparision
    # e = np.sort(e)  # Sort eigenvalues in increasing order
    # E0 = np.sum(e[0:Nocc])  # Sum over the lowest Nocc states using absurd indexing

    #### INITIAL IN-PLACE LEARNING FOR FIRST LAYER
    (hN, h1) = gershgorin(X0)  # Alternatively, obtain eigenvalue estimates using Gersgorin circle theorem
    W0 = -1 / (hN - h1)  # Weight (scalar)
    B0 = (hN / (hN - h1)) * I  # Bias (diagonal matrix)

    #### INITIAL LINEAR TRANSFORM
    S0 = W0 * X0 + B0
    S0 = np.single(S0)  # Store in single precision
    TrS0 = np.trace(S0)  # Keep track of occupation

    if dm_only == False:
        S1 = W0 * X1
        S1 = np.single(S1)  # Store in single precision
        TrS1 = np.trace(S1)  # Keep track of occupation

    start = time.time()
    #### COMPUTATIONAL DEEP LAYERS
    for layer in range(maxlayer):
        """ SP2 """
        #### ACTIVATION FUNCTION FROM TWO DUAL HALF-PRECISION MATRIX-MATRIX MULTIPLICATIONS
        X0_h = np.single(
            np.half(S0)
        )  # First half-precision repsentation of X, single used to allow single accumulation
        X0_l = np.single(
            np.half(S0 - X0_h)
        )  # Second half-precision repsentation of X, single used to allow single accumulation
        X0_hh = np.single(np.matmul(X0_h, X0_h))  # Half-precision multiplication with single accumulation
        X0_hl = np.single(np.matmul(X0_h, X0_l))  # Half-precision multiplication with single accumulation
        X0_lh = np.transpose(X0_hl)  # Use the matrix symmetry of X0 and X1 from the symmetry of S
        X0 = np.single(X0_hh + X0_hl + X0_lh)  # Additions in single precision
        TrX0 = np.trace(X0)  # Approximate occupation
        print("TrX0",TrX0)
        """""" """"""

        """ response """
        if dm_only == False:
            #### ACTIVATION FUNCTION FROM TWO DUAL HALF-PRECISION MATRIX-MATRIX MULTIPLICATIONS
            X1_h = np.single(
                np.half(S1)
            )  # First half-precision repsentation of X, single used to allow single accumulation
            X1_l = np.single(
                np.half(S1 - X1_h)
            )  # Second half-precision repsentation of X, single used to allow single accumulation
            X0X1_hh = np.single(np.matmul(X0_h, X1_h))  # Half-precision multiplication with single accumulation
            X0X1_hl = np.single(np.matmul(X1_h, X0_l))  # Half-precision multiplication with single accumulation
            X0X1_lh = np.single(np.matmul(X1_h, X0_l))  # Use the matrix symmetry of X0 and X1 from the symmetry of S
            X0X1 = np.single(X0X1_hh + X0X1_hl + X0X1_lh)  # Additions in single precision
            X1X0 = np.transpose(X0X1)
            X1 = np.single(X0X1 + X1X0)
            TrX1 = np.trace(X1)  # Approximate occupation
        """""" """"""

        #### ERROR ESTIMATE OF IDEMPOTENCY
        idemp_err[layer] = TrS0 - TrX0  # Error estimate for in-place learning and convergence control

        if dm_only == False:
            idemp_err_1[layer] = TrS1 - TrX1  # Error estimate for in-place learning and convergence control
            print(
                layer, "Idemp error estimate:" + str(idemp_err_1[layer])
            )  # Can reach 0 exactely in low precision arithmetics

        #### LEARNING THROUGH A BINARY ON-THE-FLY IN-PLACE ERROR MINIMIZATION, WHERE sgn = (+/-)*1
        sgn = np.sign(np.abs(2 * TrS0 - TrX0 - Nocc) - np.abs(TrX0 - Nocc) - sgn * eps)
        v_sgn[layer] = sgn  # Vector with the sgn to keep track
        W = sgn  # Weight function
        B = (1 - sgn) * S0  # Bias function

        #### LINEAR TRANSFORM
        S0 = W * X0 + B  # Affine linear transform, apply weight and bias
        if dm_only == False:
            S1 = W * X1 + (1 - W) * S1  # Affine linear transform, apply weight and bias

        #### KEEP TRACK OF THE NEW OCCUPATION
        TrS0 = W * TrX0 + (1 - sgn) * TrS0  # Update trace
        if dm_only == False:
            TrS1 = W * TrX1 + (1 - sgn) * TrS1  # Update trace

        #### CONVERGENCE TEST
        if idemp_err[layer] <= 0:
            break
        if (
            layer > 1
            and v_sgn[layer - 1] != v_sgn[layer - 2]
            and idemp_err[layer] >= Csp2 * idemp_err[layer - 2] * idemp_err[layer - 2]
        ):
            break

    #### POST-PROCESSING REFINEMENT STEP
    if refi == True:
        #### WITH ACTIVATION FUNCTION REFINEMENT, f(X) = 2*X^2 - X^4, IN DOUBLE PRECISION
        X = np.double(S)
        X = 2 * X - np.matmul(X, X)
        TrS = np.trace(X)
        X = np.double(np.matmul(X, X))
        TrX = np.trace(X)
        idemp_err[layer + 1] = TrS - TrX
        print(layer + 1, "Refined error estimate = " + str(idemp_err[layer + 1]))
        num_deep_layers = layer + 1  # +1 to account for the last refinement layer
        D = X  # Output layer estimate of density matrix
    else:
        #### WITHOUT ACTIVATION FUNCTION REFINEMENT, f(X) = X      # Or, alternativley use half-precision multiplications
        num_deep_layers = layer
        D0 = np.double(S0)  # Output layer estimate of density matrix D
        if dm_only == False:
            D1 = np.double(S1)  # Output layer estimate of density matrix D

    end = time.time()
    flops = 2 * N * N * N * num_deep_layers / (end - start)
    print(str(end - start) + " sec")
    print("FLOPS = " + str(flops / 1e12) + " teraflops")

    #### DOUBLE PRECISION ERROR ESTIMATES OF THE CONVERGED DENSITY MATRIX D
    occ_err = np.abs(np.trace(D0) - Nocc)  # Occupation Error
    D02 = np.matmul(D0, D0)  # Matrix square (for error analysis only!)
    idem_err = np.abs(np.trace(D02) - np.trace(D0))  # Error estimate, Tr(X2-X)
    idem_err_2_norm = np.linalg.norm(D02 - D0)  # Idempotency Error in 2-norm
    comm_err = np.linalg.norm(np.matmul(D0, H0) - np.matmul(H0, D0))  # Commutation error
    energy = np.trace(-np.matmul(D0, H0))  # Band-energy
    # energy_err = np.abs(energy - E0)  # Band-energy error

    if not dm_only:
        energy_1 = np.trace(np.matmul(D0, H1))
        energy_2 = 0.5 * np.trace(np.matmul(D1, H1))
    else:
        energy_1 = None
        energy_2 = None

    if dm_only:
        return D0
    else:
        return D0, D1
