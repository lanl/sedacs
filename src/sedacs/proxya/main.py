import numpy as np
from sedacs.system import *
from proxy_a import *
from sedacs.periodic_table import ptable

pt = ptable()
latticeVectors,symbols,types,coords = read_pdb_file("coords.pdb",lib="None",verb=True)
nats = len(coords[:,1])
write_xyz_coordinates(coords,types,symbols)
H = get_hamiltonian(coords)
N = len(H[:,1])
Nocc = int(N/2)
D = get_densityMatrix(H,Nocc)
print("Hamiltonian = ",H)
print("Density Matrix = ",D)
