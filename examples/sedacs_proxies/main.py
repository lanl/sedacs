from sedacs.io import read_pdb_file, write_xyz_coordinates
from sedacs.periodic_table import PeriodicTable

from proxies.python.first_level import get_hamiltonian

pt = PeriodicTable()
latticeVectors, symbols, types, coords = read_pdb_file("coords.pdb", lib="None", verb=True)
nats = len(coords[:, 1])
write_xyz_coordinates(coords, types, symbols)
H = get_hamiltonian(coords)
N = len(H[:, 1])
Nocc = int(N / 2)
D = get_densityMatrix(H, Nocc)
print("Hamiltonian = ", H)
print("Density Matrix = ", D)
