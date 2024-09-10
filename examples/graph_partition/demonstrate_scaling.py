from sedacs.graph_partition2 import coords_partition
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time

# Prints out info about params.
# print(help(coords_partition))

ks = [128, 256, 512, 1024]
ngx = 4; ngy = 8; ngz = 8
ngs = [[4,4,8], [4,8,8], [8,8,8], [8,8,16]]
times = []

for k, ng in zip(ks, ngs):
    print(ng)
    ngx = ng[0]
    ngy = ng[1]
    ngz = ng[2]
    t = time.time()
    partition_K_core, partition_K_halo, partition_K_num_core, partition_K_num_halo = coords_partition(
        "TrPCage_wrapped.xyz",
        # "water_10k.xyz",
        k,
        [ngx, ngy, ngz],
        device="cpu",
        cutoff=2.4,
        numSwapRuns=4,
        numMitRuns=4,
        visualize=False
    )
    t2 = time.time()
    times.append(t2-t)
    print(ks, times)

