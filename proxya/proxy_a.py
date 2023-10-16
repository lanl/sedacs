import numpy as np
import scipy.linalg as sp

def get_hamiltonian(coords,atomTypes=np.zeros((1),dtype=int)):

  """Construct simple toy s-Hamiltonian """

  N = len(coords[:,1]); Nocc = int(N/4); eps = 1e-9; decay_min = 0.1; m = 78;
  a = 3.817632; c = 0.816371; x = 1.029769; n = 13;
  b = 1.927947; d = 3.386142; y = 2.135545;
  H = np.zeros((N,N)); xx = np.zeros(N); yy = np.zeros(N);
# Construct quasi-randomized Hamiltonian H for the full system
  cnt = 0 
  R = np.zeros((N,1))
  for i in range(0,N):  # Actually from 0 to N-1
    x = (a*x+c)%m       # Hamiltonian parameters
    y = (b*y+d)%n      
    xx[i] = x ; yy[i] = y;
    for j in range(i,N): # Actually from i to N-1
      dist = np.linalg.norm(coords[i,:]-coords[j,:])
      tmp = (x/m)*np.exp(-(y/n + decay_min)*(dist**2))
      H[i,j] = tmp
      H[j,i] = tmp
   #end
  #end
  return H 

def get_densityMatrix(H,N,Nocc):
  
  """Calcualted the full density matrix from H"""

  E,Q = sp.eigh(H)
  mu = 0.5*(E[Nocc] + E[Nocc + 1])
  D = np.zeros((N,N))
  print("Q=",Q)
  for i in range(0,N):
    if (E[i] < mu):
      D = D + np.outer(Q[:,i],Q[:,i])
      print(i,"D=",D)
      print(i,"Q=",Q[:,1])
    #endif
  #endfor
  print("mu = ",mu)
  print("E=",E)
  return D

if(__name__ == '__main__'):

  coords = np.zeros((4,3))
  coords[0,0] = 2.3; coords[0,1] = -.12; coords[0,2] = 5.1;
  coords[1,0] = 2.1; coords[1,1] = -1.12; coords[1,2] = 3.3;
  coords[2,0] = 1.2; coords[2,1] = -3.32; coords[2,2] = 1.2;
  coords[3,0] = -0.2; coords[3,1] = -2.12; coords[3,2] = 2.7;

  H = get_hamiltonian(coords)
  print("H=",H)
  D = get_densityMatrix(H,4,2)
  print("D=",D)
  
