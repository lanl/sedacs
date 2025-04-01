#!/usr/bin/env python
import numpy as np
import os, sys, argparse

def generate_H(N):
    #initialize randomized symmetric test Hamiltonian
    H=np.zeros((N,N))

    for i in range(0,N):
        for j in range(i,N):
            H[i,j] = np.single(np.single(np.exp(-.5*np.abs(i-j)))*np.single(np.sin(np.single(i+1))))
            H[j,i] = H[i,j]
    return H


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



    H = generate_H(N)
    cpp_mat = np.genfromtxt('mat.csv',delimiter=',')
    
    ##### Refinement in double precision
    cpp_mat = np.matmul(cpp_mat,cpp_mat)
    D_cuda = 2*cpp_mat-np.matmul(cpp_mat,cpp_mat)
    
    ###### Calculate and print final errors
    print("cuda Occupation error:" + str(np.abs(np.trace(D_cuda)-Nocc)))
    print("cuda Idempotency error:" + str(np.linalg.norm(np.matmul(D_cuda,D_cuda)-D_cuda)))
    print("cuda Commutation error:" + str(np.trace(np.matmul(D_cuda,H)-np.matmul(H,D_cuda))))
    print("cuda Energy:" + str(np.trace(np.matmul(D_cuda,H))))
