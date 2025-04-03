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
  - - This should be handled by the main.py script.
  - `pyseqm_path`
  - - If installing pyseqm with pip, no modification to the Python import path is needed.
- In `input.in`, modify the paths for:
  - `Path`

#### Environment and Execution

1. Set the desired number of OpenMP threads:
   ```shell
   export OMP_NUM_THREADS=4
   ```
2. Activate the required environment:
   ```shell
   cd JOB_DIRECTORY
   source activate sedacs
   ```
3. Run the calculation on two ranks:
   ```shell
   mpirun -bind-to none -n 2 python -u main.py > out.out 2>&1
   ```

#### Partitioning

- The system (e.g., nanostar) will be partitioned into parts as specified by the `NumParts` keyword in `input.in`.
- Each rank will process an equal number of parts. For example, with `NumParts= 4` and `-n 2`, each rank processes 2 parts.
- **Important:** Ensure `NumParts` is divisible by the number of ranks specified with `-n`.

---

### GPU Setup and Execution

#### File Modifications

- In `input.in`, set:
  - `scfDevice= cuda`
  - If running on a single machine, set `numGPU= -1` (the code will detect the number of available GPUs automatically).

#### Environment and Execution

1. Set the desired number of OpenMP threads:
   ```shell
   export OMP_NUM_THREADS=4
   ```
2. Activate the required environment:
   ```shell
   cd JOB_DIRECTORY
   source activate sedacs
   ```
3. Run the calculation on two ranks:
   ```shell
   mpirun -bind-to none -n 2 python -u main.py > out.out 2>&1
   ```

#### Partitioning and GPU Utilization

- The system will be partitioned into parts as specified by the `NumParts` keyword.
- Partitioned parts are distributed across GPUs. If only one GPU is available, all parts will be processed sequentially on rank 0.
- **Hybrid GPU-CPU Execution:**
  - Hamiltonian construction, diagonalization, energy/forces calculations are performed on GPUs.
  - Density matrix updates are performed on the CPU, in parallel on all ranks specified with `-n`. For example, with `-n 4` and `NumParts= 4`, the i-th rank updates the portion of the density matrix corresponding to i-th part (aka i-th core+halo).
  - Graph updates and density matrix contraction are performed on rank 0.
- **Important:** Ensure `NumParts` is divisible by the number of ranks (`-n`) and the number of available GPUs. The number of ranks (`-n`) must be `>=` the number of available GPUs. 

---

### Notes

- Ensure all required paths and parameters are set correctly before running the calculation.
- Use `NumParts` carefully to optimize performance based on your available hardware (ranks and GPUs).

