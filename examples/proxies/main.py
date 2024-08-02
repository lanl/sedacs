from sedacs.file_io import read_pdb_file, write_xyz_coordinates
from sedacs.periodic_table import PeriodicTable

from proxies.python.first_level import get_density_matrix, get_hamiltonian_proxy

pt = PeriodicTable()
latticeVectors, symbols, types, coords = read_pdb_file("coords.pdb", lib="None", verb=True)
nats = len(coords[:, 1])
write_xyz_coordinates("proxy.xyz", coords, types, symbols)
H = get_hamiltonian_proxy(coords)
N = len(H[:, 1])
Nocc = int(N / 2)
D = get_density_matrix(H, Nocc)
print("Hamiltonian = ", H)
print("Density Matrix = ", D)
