from mpi4py import MPI
import numpy as np
import time
import torch
#numDataPerRank = torch.zeros((20000, 20000), dtype=torch.float64) + 2.22

#torch.set_num_threads(8)  
tic = time.perf_counter()

numDataPerRank = torch.zeros((30000, 30000), dtype=torch.float64) + 2.22
#numDataPerRank = np.zeros((30000, 30000), dtype='d') + 2.22
for i in numDataPerRank:
    i *= 1.925

print("Time", time.perf_counter() - tic,"(s)")
print(numDataPerRank.device)
#print(torch.get_num_threads())  # Check the number of threads


