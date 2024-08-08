from sedacs.neighbor_list import NeighborState
from sedacs.io import read_coords_file

coords_file = "water_6540.pdb"
lattice_vecs, symbols, types, coords = read_coords_file(coords_file, lib="None", verb=True)