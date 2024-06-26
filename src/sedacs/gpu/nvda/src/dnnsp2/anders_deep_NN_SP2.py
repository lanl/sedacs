#!/usr/bin/env python
import numpy as np
import os,sys,argparse,time

def generate_H(N):
    #initialize randomized symmetric test Hamiltonian
    H=np.zeros((N,N))

    for i in range(0,N):
        for j in range(i,N):
            H[i,j] = np.exp(-.5*np.abs(i-j))*np.sin(i+1);
            H[j,i] = H[i,j];

    return H

def gershgorin(M):
    #find eigenvalue estimates of the matrix M from the Gershgorin circle theorem
    min_e=0
    max_e=0

    for i in range(0,np.shape(M)[0]):
        e=M[i,i] #Gershgorin eigenvalue circle center
        r=0

        for j in range(0,np.shape(M)[0]):  #compute sum of abs. val of components in row i
           r+=np.abs(M[i,j])

        r-=np.abs(e) #Gershgorin eigenvalue circle radius

        #update min and max eigenvalues as you loop over rows
        if e-r < min_e:
            min_e = e-r
        elif e+r > max_e:
            max_e = e+r

    return (min_e,max_e)

def dual_half(S):
    # Calculation of X^2 as defined in equation (15). Note that precision in numpy is closed under alegbraic operations when of the same data type and inherits the precision of the highest precision operand when not of the same data type.
    S0=np.half(S)
    S1=np.single(np.half(S-S0))
    S0S0 = np.single(np.matmul(S0,S0))
    S0S1 = np.matmul(S0,S1)
    X = S0S0 + (S0S1 + np.transpose(S0S1))
    return X

#### DEEP-NN FORMULATION OF THE RECURSIVE SP2 FERMI-OPERATOR EXPANSION SCHEME
if __name__=="__main__":  #execute main script only when directly called from the command line
    np.set_printoptions(precision=15)
    ##### Input initilization values
    parser = \
            argparse.ArgumentParser(description='Input N, number of electrons, and Nocc, the occupation number.', \
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-n',
                      dest='N',
                      type=int,
                      help='number of electrons',
                      required=True)
    parser.add_argument('-nocc',
                      dest='Nocc',
                      type=int,
                      help='number of occupied orbitals',
                      required=True)

    args = parser.parse_args()
    N = args.N
    Nocc = args.Nocc


    #### INITIALIZE
    eps = 1e-16                     # Small value, but such that +/- eps is finite in single precision
    Csp2 = 4.5                      # Convergence criterion as derived by Emanuel 
    sgn = 0                         # Initial value of sgn
    I = np.eye(N)                   # Identity matrix
    maxlayer = 100                  # Maximum number of layers
    v_sgn = np.zeros(maxlayer)      # Keeps track of binary in-place learning choices
    idemp_err = np.zeros(maxlayer)  # Local error estimate
    idemp_err_1 = np.zeros(maxlayer)  # Local error estimate

    #### CHOSE POST-PROCESSING ACTIVATION FUNCTION REFINEMENT OR NOT
    Refinement = True        # Or False

    #### LOAD/CONSTRUCT HAMILTONIAN AS INPUT LAYER
    H0 = generate_H(N)        # Symmetric NxN Hamiltonian matrix
    X0 = H0                   # Initial input layer
    H1 = 0.1*I                   # Initial input layer
    X1 = H1
    #### 'EXACT' SOLUTION FOR COMPARISION ONLY
    e,v = np.linalg.eig(H0+H1)   # Diagonlize H as a brute force comparision
    e = np.sort(e)           # Sort eigenvalues in increasing order
    E0 = np.sum(e[0:Nocc])   # Sum over the lowest Nocc states using absurd indexing

    #### INITIAL IN-PLACE LEARNING FOR FIRST LAYER
    h1 = -1.867              # Some prior estimate of lower spectral bound (e[0]) for specific test problem 
    hN = 1.867               # Some prior estimate of upper spectral bound (e[N]) for specific test problem
    (hN,h1) = gershgorin(X0+X1)   # Alternatively, obtain eigenvalue estimates using Gersgorin circle theorem
    W0 = -1/(hN-h1)          # Weight (scalar)
    B0 = (hN/(hN-h1))*I      # Bias (diagonal matrix)

    #### INITIAL LINEAR TRANSFORM
    S0 = W0*X0 + B0
    S0 = np.single(S0)         # Store in single precision
    TrS0 = np.trace(S0)        # Keep track of occupation
    
    S1 = W0*X1 
    S1 = np.single(S1)         # Store in single precision
    TrS1 = np.trace(S1)        # Keep track of occupation
    
    start=time.time()
    #### COMPUTATIONAL DEEP LAYERS
    for layer in range(maxlayer):

        #### ACTIVATION FUNCTION FROM TWO DUAL HALF-PRECISION MATRIX-MATRIX MULTIPLICATIONS
        X0_h = np.single(np.half(S0))         # First half-precision repsentation of X, single used to allow single accumulation
        X0_l = np.single(np.half(S0-X0_h))      # Second half-precision repsentation of X, single used to allow single accumulation
        X0_hh = np.single(np.matmul(X0_h,X0_h)) # Half-precision multiplication with single accumulation
        X0_hl = np.single(np.matmul(X0_h,X0_l)) # Half-precision multiplication with single accumulation
        X0_lh = np.transpose(X0_hl)          # Use the matrix symmetry of X0 and X1 from the symmetry of S
        X0 = np.single(X0_hh + X0_hl + X0_lh)  # Additions in single precision 
        TrX0 = np.trace(X0)                  # Approximate occupation

        #### ACTIVATION FUNCTION FROM TWO DUAL HALF-PRECISION MATRIX-MATRIX MULTIPLICATIONS
        X1_h = np.single(np.half(S1))         # First half-precision repsentation of X, single used to allow single accumulation
        X1_l = np.single(np.half(S1-X1_h))      # Second half-precision repsentation of X, single used to allow single accumulation
        X0X1_hh = np.single(np.matmul(X0_h,X1_h)) # Half-precision multiplication with single accumulation
        X0X1_hl = np.single(np.matmul(X1_h,X0_l)) # Half-precision multiplication with single accumulation
        X0X1_lh = np.single(np.matmul(X1_h,X0_l))  # Use the matrix symmetry of X0 and X1 from the symmetry of S
        X0X1 = np.single(X0X1_hh + X0X1_hl + X0X1_lh)  # Additions in single precision 
        X1X0 = np.transpose(X0X1)
        X1 = np.single(X0X1 + X1X0)
        TrX1 = np.trace(X1)                  # Approximate occupation
        
        
        #### ERROR ESTIMATE OF IDEMPOTENCY
        idemp_err[layer] = TrS0-TrX0                               # Error estimate for in-place learning and convergence control
        idemp_err_1[layer] = TrS1-TrX1                               # Error estimate for in-place learning and convergence control
        #print(layer, "Idemp error estimate:" + str(idemp_err[layer]))  # Can reach 0 exactely in low precision arithmetics
        print(layer, "Idemp error estimate:" + str(idemp_err_1[layer]))  # Can reach 0 exactely in low precision arithmetics



        #### LEARNING THROUGH A BINARY ON-THE-FLY IN-PLACE ERROR MINIMIZATION, WHERE sgn = (+/-)*1
        sgn = np.sign(np.abs(2*TrS0 - TrX0 - Nocc) - np.abs(TrX0 - Nocc) - sgn*eps)      
        v_sgn[layer] = sgn         # Vector with the sgn to keep track
        W = sgn                    # Weight function
        B = (1-sgn)*S0              # Bias function

        #### LINEAR TRANSFORM
        S0 = W*X0 + B                # Affine linear transform, apply weight and bias 
        S1 = W*X1 + (1-W)*S1                # Affine linear transform, apply weight and bias 

        #### KEEP TRACK OF THE NEW OCCUPATION
        TrS0 = W*TrX0 + (1-sgn)*TrS0  # Update trace
        TrS1 = W*TrX1 + (1-sgn)*TrS1  # Update trace

        #### CONVERGENCE TEST
        if (idemp_err[layer] <= 0):
          break
        if layer > 1 and v_sgn[layer-1] != v_sgn[layer-2] and idemp_err[layer] >= Csp2*idemp_err[layer-2]*idemp_err[layer-2]:
          break


    Refinement = 0
    #### POST-PROCESSING REFINEMENT STEP
#    if Refinement:
#      #### WITH ACTIVATION FUNCTION REFINEMENT, f(X) = 2*X^2 - X^4, IN DOUBLE PRECISION
#      X = np.double(S)
#      X = 2*X-np.matmul(X,X)             
#      TrS = np.trace(X)
#      X = np.double(np.matmul(X,X))          
#      TrX = np.trace(X)
#      idemp_err[layer+1] = TrS-TrX
#      print(layer+1,'Refined error estimate = ' + str(idemp_err[layer+1]))
#      num_deep_layers = layer+1                                  # +1 to account for the last refinement layer
#      D = X                                                      # Output layer estimate of density matrix
#    else:
      #### WITHOUT ACTIVATION FUNCTION REFINEMENT, f(X) = X      # Or, alternativley use half-precision multiplications
    num_deep_layers = layer                                    
    D0 = np.double(S0)                                           # Output layer estimate of density matrix D 
    D1 = np.double(S1)                                           # Output layer estimate of density matrix D 
    end=time.time()
    flops = 2*N*N*N*num_deep_layers/(end-start)
    print(str(end-start) + " sec")
    print("FLOPS = " + str(flops/1e12) + " teraflops")
    f = open("python_timings.csv","a+")
    f.write("%d, %f\r\n" % (N,flops))
    f.close()

    #### DOUBLE PRECISION ERROR ESTIMATES OF THE CONVERGED DENSITY MATRIX D 
    occ_err = np.abs(np.trace(D0)-Nocc)                           # Occupation Error
    D02 = np.matmul(D0,D0)                                          # Matrix square (for error analysis only!)
    idem_err = np.abs(np.trace(D02) - np.trace(D0))                # Error estimate, Tr(X2-X)
    idem_err_2_norm = np.linalg.norm(D02 - D0)                     # Idempotency Error in 2-norm
    comm_err = np.linalg.norm(np.matmul(D0,H0)-np.matmul(H0,D0))     # Commutation error
    energy = np.trace(-np.matmul(D0,H0))                            # Band-energy 
    energy_err = np.abs(energy-E0)                               # Band-energy error

    energy_1 = np.trace(np.matmul(D0,H1))
    energy_2 = 0.5*np.trace(np.matmul(D1,H1))

    print(' ')
    print('Final double-precision error analysis:')
    print("Number of layers:" + str(num_deep_layers))
    print("Occupation error:" + str(occ_err))
    print("Idempotency trace estimate:" + str(idem_err))
    print("Idempotency Error in 2-norm:" + str(idem_err_2_norm))
    print("Commutation 2-norm error:" + str(comm_err))
    print("Band-energy error:" + str(energy_err))
    print("Relative energy error:" + str(energy_err/energy))
    print("Calculated zero-th order energy:" + str(energy))
    print("Calculated first order energy:" + str(energy_1))
    print("Calculated second order energy:" + str(energy_2))
    print('Exact energy = ' + str(E0))
