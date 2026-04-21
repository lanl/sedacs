#!/bin/bash
#SBATCH --job-name=sedacs
#SBATCH -A mXXXX
#SBATCH -C gpu
#SBATCH -q regular 
#SBATCH -t 0:59:00
#SBATCH -N 1 
#SBATCH --ntasks-per-node=16
#SBATCH -c 8
#SBATCH --output=Out.%j
#SBATCH --error=Err.%j
#SBATCH --gpus-per-node=4

module load python
mamba activate $HOME/mamba_sedacs 
export LATTE_PATH=~/LATTE/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/bml/install/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/qmd-progress/install/lib64/
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_NUM_THREADS=4

srun --cpu-bind=cores -n 16 -c 8 ./select_gpu_device python main.py --md_iter 100 
exit
