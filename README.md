# generalCodes

### Example Mac conda environment with MPS acceleration
```
conda create -n sedacs  pytorch::pytorch torchvision torchaudio openmpi mpi4py scipy jupyter nb_conda_kernels python=3.10 -c pytorch
```
### Example Linux conda environment with CUDA acceleration
```
conda create -n sedacs torchvision torchaudio pytorch-cuda=11.8 openmpi mpi4py scipy jupyter nb_conda_kernels python=3.10 -c pytorch -c nvidia
```

## proxya

Proxya code as explained in the proposal. This proxy code a should 
perform up to a full SCF optimization of the DM. Written in 
three different languages: python, fortran, and C. 

## gpu
This is an early implementation of the gpu/AI-solver library.

## latte 

This is just a code that generates "Latte" Hamiltonians from a coordinates
input file (`xyz` or `pdb`) and constructs the DM. 

## mods

Auxyliary python modules. These modules will be used as bulding blocks 
to develop SEDACS.

## driver

Scripts to exercise the code


