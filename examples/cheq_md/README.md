# Charge Equilibration Example

This script performs molecular dynamics (MD) simulations with an optional long-range electrostatics treatment (Ewald or PME) and electronegativity equalization (QEQ). It uses a neural-network-based short-range model (via `hippynn`) combined with a QEQ solver for partial charges and optionally includes Shadow MD integration to improve long-term energy conservation.

## Features

1. **Short-range interactions**  
   - Modeled by a neural network (`hippynn` framework).  
   - Neighbor lists are constructed (based on a specified cutoff) for efficient short-range force calculations.

2. **Long-range interactions** (Ewald or Particle Mesh Ewald/PME)  
   - Ewald summation or PME for electrostatics.  
   - Includes QEQ solver for partial charge assignment.

3. **Shadow MD** (Optional)  
   - An integration scheme that leverages a “shadow potential” to enhance stability.  
   - Controlled with the `--use_shadow` argument.

4. **Integrator Types**  
   - NVE: microcanonical ensemble (constant energy).  
   - NVT: canonical ensemble (constant temperature) with Langevin thermostat (`--is_nvt`).


## Dependencies

Make sure the following Python packages (and their dependencies) are installed:

- [torch (PyTorch)](https://pytorch.org/)  
- [numpy](https://numpy.org/)  
- [matplotlib](https://matplotlib.org/)  
- [tqdm](https://github.com/tqdm/tqdm)  
- [ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/)  
- [hippynn](https://github.com/lanl/hippynn)  

## Usage

Run the script from the command line with arguments:

```bash
python run_MD.py [args...]
```

**Key Arguments:**

- **`--model_folder`** *(str, default="model_data")*  
  Path to the trained Hippynn model folder.

- **`--dt`** *(float, default=0.4)*  
  Time step for integration in fs.

- **`--N`** *(int, default=6540)*  
  System size (number of atoms). Valid choices: `6540, 10008, 25050, 52320`.

- **`--use_ewald`** *(int, default=1)*  
  Whether or not to use long-range Ewald/PME. `1` = use, `0` = ignore.

- **`--use_shadow`** *(int, default=1)*  
  Enables or disables Shadow MD. `1` = enable, `0` = disable.

- **`--rtol`** *(float, default=1e-1)*  
  Convergence tolerance for the QEQ solver.

- **`--prefix`** *(str, default="MD")*  
  Prefix for output files.

- **`--sim_length`** *(int, default=200)*  
  Total simulation time in fs.

- **`--is_nvt`** *(int, default=0)*  
  `1` = NVT simulation, `0` = NVE simulation.

- **`--restart`** *(int, default=0)*  
  Whether to start from a restart file (`.pt`). Must exist if enabled.

- **`--cutoff`** *(float, default=10.0)*  
  Long-range cutoff distance (Å).

- **`--buffer`** *(float, default=1.0)*  
  Buffer for neighbor list construction (Å).

- **`--ewald_method`** *(`Ewald` or `PME`, default="PME")*  
  Select the long-range summation method.

- **`--ewald_err`** *(float, default=5e-4)*  
  Desired accuracy for the Ewald/PME force.

- **`--precision`** *(`FP32` or `FP64`, default="FP64")*  
  Select floating-point precision.

- **`--use_jacobi_precond`** *(int, default=0)*  
  Use a Jacobi preconditioner for the QEQ solver if set to `1`.

- **`--dx`, `--dy`, `--dz`** *(int, default=1)*  
  Replicate the system in x, y, or z directions.

- **`--plot_results`** *(int, default=1)*  
  Whether to generate basic plots of the results at the end.

### Example Command

```bash
python run_MD.py \
  --model_folder="model_data" \
  --dt=0.2 \
  --N=6540 \
  --use_shadow=1 \
  --rtol=1e-1 \
  --prefix="MD_run" \
  --sim_length=10000 \
  --is_nvt=0 \
  --cutoff=10.0 \
  --buffer=1.0 \
  --ewald_method="PME" \
  --ewald_err=5e-4 \
  --precision="FP64" \
  --plot_results=1
```

## Outputs

1. **Simulation Trajectory / Final Structures**  
   - `*_final.pdb`: the final structure after simulation.  
   - `*_system.pt`: a PyTorch-saved `SystemState`, which can be used for restart.

2. **Energies & Temperatures**
   - `*_potential_energy.txt`: Potential energies over time.  
   - `*_temp.txt`: Temperatures over time.  
   - `*_total_energy.txt`: Total energies over time.  

3. **Partial Charges & Positions**
   - `*_all_charges.npy`: Array of partial charges per snapshot.  
   - `*_all_positions.npy`: Array of positions per snapshot.

4. **Timing Data**  
   - `*_timing.txt`: Breakdown of solver time, neighbor-list updates, MatVec calls, etc.

5. **PDF Plots**
   - `*_potential_energy.pdf`: Potential energy vs. time.  
   - `*_temp.pdf`: Temperature vs. time.  
   - `*_total_energy.pdf`: Total energy vs. time.