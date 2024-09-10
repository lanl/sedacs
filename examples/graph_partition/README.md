# Graph Partitioning routine.
- (TODO remove `graph_partition.py` and replace with `graph_partition2.py`)

## Files in the current folder are as follows:
- `demonstrate_scaling.py`
- - Demonstrates the scaling of the initial coordinates based partitioning. This serves the purpose of checking speedups to the code.
- - ***For systems of ~30k nodes, this is way too expensive.*** (see water_10k.xyz as an input file)
- - TODO: Integrate new neighborlist code for initial spatial partitioning.
- - TODO: Remove nNodes^2 scaling in favor of nNode.nParts scaling as the all-to-all approach may be redundant and extremely expensive for large systems.

- `run_mitigation.py`
- - Runs the initial partitioning based on coordinates. Serves as an example for generating an initial node partitioning from the geometric data.
- - Runs the swap routine (where nodes are compared pairwise and considered for partition exchange)
- - Runs the mitigation routine (where nodes from large partitions are considered for swaps to smaller partitions to mitigate the large C+H partitions.

- `run_regular_mincut.py`
- - Both run the routines for getting a regular partitioning (e.g. simple grouping of nodes into k desired partitions) and a minucut partitioning (one k-fold partitioning that minimizes the cut)

## Useful functions in sedacs.graph_partition:

### `partition_from_coordinates`
The main function to use here is sedacs.graph_partition2.partition_from_coordinates 

Import the function as:
`from sedacs.graph_partition2 import partition_from_coordinates`

Docstrings on the function explain all possible arguments, use `help(partition_from_coordinates)` if extra detail is necessary.
*To be added is the routine to update the partition given new coordinates/globally threshold density matrix expansion.

A MWE of the function looks like this, and can be seen in `run_mitigation.py`:
Positional arguments:
1. Structure file name as a string
2. Number of desired partitions `k`.
3. Domain decomposition list, or tuple of length 3. `(ngx, ngy, gz)`, where their product must be a divisor of `k`.
- - E.g. (4,4,8) would partition x, y into 4 sections, and z into 8. `k` then must be 4*4*8 = 128
- - In the future we probably want a helper function to determine a reasonable spatial partitioning for the user automatically if not provided.

The rest of the arguments are keyword arguments and aren't needed. The ones you might want to use are:

`device` (string, must be a torch backend)-> determines torch device for the initial partitioning scheme (though this will be swapped out for neighborlist routines)

`cutoff` (float)-> Cutoff in the same units as the structure file for determining edges between atoms.

`numSwapRuns` (int)-> Number of runs the algorithm attempts to swap node partitions to get more balance and fewer cuts.

`numMitRuns` (int)-> Number of runs where the large partitions have node removed from them to mitigate large core+halos.
- Reasonable numbers for num_swap_runs and num_mit_runs are ~5-10. num_mit_runs is a less expensive routine, but relies on a decent partitioning already existing from the swap runs.
`visualize` (boolean) -> Turn this off when using in production, but it provides a helpful visualization to make sure the partitioning looks decent.
`verbosity` (int) -> Set this to 0 to turn off information being printed at each `swap` and `mit` iteration.


The function then returns four things:
1. partitionKCore 
2. partitionKHalo 
3. partitionKNumCore 
4. partitionKNumHalo 

The first two returns contain the actual data regarding which the core nodes/halos of partition K. These are padded numpy arrays. 

Because these are padded numpy arrays, we return also how many core/halo nodes are in each partition.

In other words, to get the K=5th partitions core nodes we'd do:
`partition_K_core[5,:partition_K_num_core[5]]` to remove the padding.

Likewise to get the halo:
`partition_K_halo[5,:partition_K_num_halo[5]]`

It may be desirable in the future to return these as unpadded lists so that we can simply do something like 

K_core_plus_halo = `partition_K_core[5] + partition_K_halo[5]`


### (TODO add graph_partition.update_partition(previousPartitioning, graph, electronicStrutureInfo)
This will be the function to take the previous partiion, as well as the electronic structure information (e.g. globally thresholded density matrix expansion), we can then just do a very quick update with the partition flips/mitigation routines to get an updated graph partitioning reflecting the new electronic structure at the given MD time step.

### `graph_partition(graph, partitionType, nparts, verb=False)`
- Wrapper for the metis, regular, and mincut partitioning routines.

1. `metis_partition(graph, nparts, verb=False)`
- Runs the partitioning with metis.

2. `mincut_partition(graph, nparts, verb, numSwapRuns = 20)`
- Runs a mincut partitioning. See `run_regular_mincut.py`

3. `regular_partition(graph, nparts, verb=False)`
- Runs a regular partitioning whereby nodes are trivially grouped.

### Utility functions for pulling graph/partitioning info.

1. `get_cut(nodeIPartition, graph):`
- Returns cut of the input graph/partiioning scheme. There may be a bug here, as input says it expects adjacency matrix, but seems not to be the case.
- The code is not currently using this function to compute the cut, but rather manually computing from the partition/graph.

2. `def get_balance_from_partition_sizes(partitionKNumNodes)`
- Computes the balance from a set of partition sizes. 
- E.g. for 4 partitions, `partitionKNumNodes` may be `[100,10,40,20]` and the balance would be `100/10`.

3. `get_balance_from_partitions(parts)`
- Takes the input partitions and computes the balance. E.g. input partitions may be a 2d list with each entry containing the k'th partition's member node indices.

4. `get_balance(nodeIPartition, k)`
- Takes an input array of nNodes length with the corresponding partition. 
- Bins the relevant info to get partition sizes and returns the balance.

5.`get_parts_indices(parts, nnodes)`
- Takes the partitions (as in `get_balance` from 4. above) and returns the array of length nNodes described in `get_balance_from_partitions in 3.
- E.g. returns something like nodeIPartition = [0, 5, 3, 2, 10] where nodeIPartition[1] = 5 => that node 1 is in partition 5.

6. `get_parts_from_indices(whichPart, nparts)`
- Inverse of the function described in 5.

