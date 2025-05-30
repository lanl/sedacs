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
│   └── source
├── examples
│   ├── cheq_md
│   ├── graph_adaptive_dm_scf_sp2
│   ├── latte_interface
│   └── pyseqm_interface
├── parameters
│   ├── fitting
│   ├── latte
│   └── siesta_calculations
├── proxies
│   ├── c
│   ├── fortran
│   ├── matlab
│   ├── matlab_proxyA
│   ├── matlab_proxyB
│   └── python
├── src
│   └── sedacs
│       ├── cheq
│       ├── dev
│       ├── driver
│       ├── ewald
│       ├── gpu
│       │   ├── amd
│       │   ├── example
│       │   ├── nvda
│       │   └── omp
│       └── interface
└── tests
```

### `src`

This folder contains the bulk of the SEDACS codebase. It is seperated into 
modules where applicable. 

### `docs`

Brief documentation for the functionality of the code. This will be replaced 
largely by auto-generated code from the docstrings throughout the code-base. 
Tentatively, the docstrings are a more up-to-date place to view the docs for 
the code.

#### `driver` 

The `driver` modules can be thought of to contain full examples of ~production
level simulations using the methods implemented in the code.

#### `ewald` 

The `ewald` folder contains tools for solving long-range interactions in 
periodic systems with the Ewald and Particle-mesh Ewald methods.

#### `cheq` 

The `cheq` folder contains the relevant code for solving chare-equilibration
models both in the standard, and Shadow/Extended-Lagrangian formalism.

#### `dev` 

The `dev` folder contains some utility functions for dealing with the paths and 
linking SEDACS to the necessary parameter sets, and/or binaries for external 
electronic structure codes in a programmatic manner.


### `proxies`

This folder holds Matlab, Python, Fortran, and C implementations of various 
proxy codes which can be thought of as quick interface for running quick 
simulations in the prototyping phase of a project. For exmample, the proxyA 
code is implemented as explained in the SEDACS proposal. This proxy code should
perform up to a full SCF optimization of the density matrix. It is written in
three different languages: Python, Fortran, and C.

### `gpu`

This is an implementation of the GPU/AI-solver library.

### `parameters`

This is a set of codes for fitting tight-binding parameters (for usage with the
LANL-developed LATTE tight-binding electronic structure code), as well as some
scripts for generating reference data with SIESTA.


# License

This program is open source under the [BSD-3 License](LICENSE.txt).
