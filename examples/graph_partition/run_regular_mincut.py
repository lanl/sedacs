from sedacs.graph_partition import mincut_partition, regular_partition
from sedacs.graph import get_random_adjacency_matrix

nNodes = 200
graphadj = get_random_adjacency_matrix(nNodes, density=0.1, degreeOnDiagonal=False)

nparts = 16
verb = 1

# Run the mincut_partition
parts = mincut_partition(graphadj, nparts, verb)

# Run the regular_partition
parts = regular_partition(graphadj, nparts, verb)

# These both return the desired node partitioning according to the selected routine.
