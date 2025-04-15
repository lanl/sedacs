#!/bin/bash
#SBATCH --job-name=sedacs
#SBATCH --output=out_%j.out
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00

# Clean environment
module purge

# Load the necessary modules on your cluster for the SEDACS + PYSEQM interface.
# This likely requires some configuration on the user's end.
module load miniconda3
module load cuda/12.0.0
source activate sedacs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Disable buffering of Python output.
export PYTHONUNBUFFERED=1

# Adapt this to your cluster's scheduling system.
srun --mpi=pmix --distribution=block --cpu-bind=none python main.py

