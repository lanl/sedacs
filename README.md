# sedacs

Scalable Ecosystem, Driver, and Analyzer for Complex Chemistry Simulations (SEDACS) enables 
massively parallel atomistic simulations that can seamlessly integrate with a diverse
range of available and emerging quantum chemistry codes at different levels
of theory. 

Supporting ab initio, semiempirical quantum mechanics (SEQM),and coarse-grained flexible charge 
equilibration (ChEQ) models, this is a unified framework to simulate and analyze
the MD of complex chemical systems and materials. 

SEDACS also enables the anlysis of trajectories using novel graph-based ML schemes 
and quantum-response information to capture and visualize hidden, non-local quantum features that cannot be seen from the geometry alone. 

Finally, SEDACS provides advanced mixed-precision electronic structure solver library
that uses AI-hardware accelerators. 

Our target customer is a Computational Chemist domain expert working on complex materials systems 
or developing  new quantum capabilities that can easily be deployed at scale. 

We hence provide transparent implementations that closely follow a “white-board” physics and mathematics presentation. The threshold to understand and work with this codebase for a domain
expert is purposely kept low. 



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

# License

This program is open source under the [BSD-3 License](LICENSE.txt).
