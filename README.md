# sedacs

## Installation

### In Mac conda environment with MPS acceleration

```shell
conda create -n sedacs  pytorch::pytorch torchvision torchaudio openmpi mpi4py scipy jupyter nb_conda_kernels python=3.10 -c pytorch
```

### In Linux conda environment with CUDA acceleration

```shell
conda create -n sedacs pytorch torchvision torchaudio pytorch-cuda=11.8 openmpi mpi4py scipy jupyter nb_conda_kernels python=3.10 -c pytorch -c nvidia
```

## Folder structure

The current codebase has the following folder structure:

```
.
├── docs
├── driver
├── gpu
├── latte
├── mods
├── proxya
└── test
```

### `proxya`

Proxya code as explained in the proposal. This proxy code should
perform up to a full SCF optimization of the density matrix. It is written in
three different languages: Python, Fortran, and C.

### `gpu`

This is an implementation of the GPU/AI-solver library.

### `latte`

This is a code that generates "Latte" Hamiltonians from input coordinates
files (`xyz` or `pdb`) and constructs the density matrix.

### `mods`

Auxiliary Python modules. These modules will be used as building blocks
to develop SEDACS.

### `driver`

Scripts to exercise the code.
