from sedacs.graph_partition import coords_partition
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
ngx = 4
ngy = 8
ngz = 8
k = 256
partition_K_core, partition_K_halo, partition_K_num_core, partition_K_num_halo = coords_partition(
    "TrPCage_wrapped.xyz",
    # "water_10k.xyz",
    k,
    [ngx, ngy, ngz],
    device="cpu",
    cutoff=2.4,
    numSwapRuns=8,
    numMitRuns=12,
    visualize=True
)
# print("Min core", np.min(partition_K_num_core))
# print("Min core+halo", np.min(partition_K_num_core + partition_K_num_halo))

# Check the correlation between the cores and core + halos to see if core is good enough proxy for mitigating large partitions.
# sns.violinplot(x=partition_K_num_core,y=partition_K_num_core + partition_K_num_halo)
# plt.xlabel("Core")
# plt.ylabel("Core+Halo")
# plt.title("Check if Core size is good proxy for Core + Halo size")
# plt.show()

# This can be used as a quick check the output looks as expected.
# core_halo_size = partition_K_num_core + partition_K_num_halo
# plt.title("Partition Sizes (Core and Core+Halo)")
# plt.ylabel("Number of nodes")
# plt.xlabel("Partition Number")
# plt.bar(list(range(k)), core_halo_size, label = "C+H", color = 'red', edgecolor = 'k')
# plt.bar(list(range(k)), partition_K_num_core,  label = "Core", color = 'blue', edgecolor = 'k')
# ch_max = np.max(core_halo_size)
# plt.yticks(np.arange(0, ch_max, int(ch_max)/8))
# plt.xlim(0, k)
# plt.legend()
# plt.show()
