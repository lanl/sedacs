## `PySEQM`

### Overview

`PySEQM` is designed to handle finite systems, which are placed in boxes with significant vacuum gaps to avoid interactions between periodic images.

### CPU Setup and Execution

#### File Modifications

- In `main.py`, update the paths for:
  - `proxya_path`
  - `pyseqm_path`
- In `input.in`, modify the paths for:
  - `Path`
  - `Executable`

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
- Each rank will process an equal number of parts. For example, with `NumParts=4` and `-n 2`, each rank processes 2 parts.
- **Important:** Ensure `NumParts` is divisible by the number of ranks specified with `-n`.

---

### GPU Setup and Execution

#### File Modifications

- In `input.in`, set:
  - `scfDevice = cuda`
  - If running on a single machine, set `numGPU = -1` (the code will detect the number of available GPUs automatically).

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
  - Density matrix updates are performed on the CPU, in parallel.
  - For example, with `-n 4` and `NumParts=4`, each rank updates one part of the density matrix.
- **Important:** Ensure `NumParts` is divisible by the number of ranks (`-n`) and the number of available GPUs.

---

### Notes

- Ensure all required paths and parameters are set correctly before running the calculation.
- Use `NumParts` carefully to optimize performance based on your available hardware (ranks and GPUs).

