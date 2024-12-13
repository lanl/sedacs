## `PySEQM`

### Overview

PySEQM is a PyTorch-based framework for Semi-Empirical Quantum Mechanics, supporting various NDDO methods such as PM6, PM3, AM1, and MNDO.
It includes built-in drivers for MD thermostats and XL-BOMD, with native support for ML (re)parametrization.
To install the latest version, use the following branch: [`PySEQM`](https://github.com/lanl/PYSEQM/tree/pm6_test).
It is designed to handle finite systems. Therefore, structures are placed in boxes with vacuum gaps to avoid interactions between periodic images.

### CPU Setup and Execution

#### File Modifications

- In `main.py`, update the paths for:
  - `proxya_path`
  - `pyseqm_path`
- In `input.in`, modify the path for:
  - `Path`

#### Environment and Execution
In slurm submission script `slurm.sl`:
1. Set the  number of nodes `--nodes`.
2. Set `--ntasks-per-node`. (`--nodes` * `--ntasks-per-node` will be the total number of ranks)
3. Set `--cpus-per-task`. This will also be used as the number of OpenMP threads.
4. Load conda and cuda modules, activate the environment.
5. (Optional) `PYTHONUNBUFFERED=1' is used for writing the output on-the fly.
6. Set your `--mpi`.

#### Partitioning

- The system (here, Gramicidin S solvated in H2O, Cl, Na, 3605 atoms) will be partitioned into 16 parts as specified by the `NumParts` keyword in `input.in`.
- Each rank will process an equal number of parts. For example, with `--nodes=2`, `--ntasks-per-node=8`, and `NumParts= 16`, each rank processes 1 part.
- **Important:** Ensure `NumParts` is divisible by `--nodes` * `--ntasks-per-node`.

---

### GPU Setup and Execution

#### File Modifications

- In `input.in`, set:
  - `scfDevice= cuda`
  - If running on a single node or nodes are homogeneous (number of GPUs per node is the same for all nodes), set `numGPU= -1` (the code will detect the number of available GPUs automatically).
  - If the number of GPUs per node is NOT the same for all nodes, set `numGPU=` to a MINIMUM number of available GPUs per node.

#### Environment and Execution

In slurm submission script `slurm.sl`:
1. Set the  number of nodes `--nodes`.
2. Set `--ntasks-per-node`. (`--nodes` * `--ntasks-per-node` will be the total number of ranks)
3. Set `--cpus-per-task`. This will also be used as the number of OpenMP threads.
4. Load conda and cuda modules, activate the environment.
5. (Optional) `PYTHONUNBUFFERED=1' is used for writing the output on-the fly.
6. Set your `--mpi`.

#### Partitioning and GPU Utilization

- The system will be partitioned into parts as specified by the `NumParts` keyword.
- Partitioned parts are evenly distributed across GPUs. If only one GPU is available, all parts will be processed sequentially on rank 0.
- **Hybrid GPU-CPU Execution:**
  - Hamiltonian construction, diagonalization, energy/forces calculations are performed on GPUs.
  - Density matrix updates are performed on the CPU, in parallel on all ranks on node 0.
  - Graph updates and density matrix contraction are performed on rank 0.
- **Important:** Ensure `NumParts` is divisible by `--nodes` * `--ntasks-per-node` and by the number of available GPUs. `--ntasks-per-node` must be `>=` the number of available GPUs per node. 

---
