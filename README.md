# SEDACS

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

We hence provide transparent implementations that closely follow a "white-board" physics and mathematics presentation. The threshold to understand and work with this codebase for a domain
expert is purposely kept low. 

---

## Installation

### Setup SEDACS–LATTE Interface

#### For rapid deployment using lighter-weight Spack + pip setup:

Install Spack
> [!NOTE] 
> Make sure you are not using the system-provided Spack or setting up other Spack beforehand in your .bashrc
```shell
git clone --depth=2 https://github.com/spack/spack.git spack_sedacs
. ~/spack_sedacs/share/spack/setup-env.sh
spack repo update
```
Clone SEDACS
```shell
git clone https://github.com/lanl/sedacs.git
```
Setting up spack environment using spack.yaml provided in this repo
```shell
spack env create sedacs sedacs/envs/latte/spack.yaml
spack env activate -p sedacs
spack concretize -f
spack install
```
Install required python dependencies using pip from spack
```shell
pip install -r sedacs/envs/latte/requirements.txt
cd sedacs
```
Install SEDACS in editable mode
```shell
pip install -e .
```
#### Full Spack build:

Build and install the complete SEDACS–LATTE stack using Spack:
```shell
spack env create sedacs sedacs/envs/latte/spack_all.yaml
spack env activate -p sedacs
spack concretize -f
spack install
```
This approach enables platform-specific optimization and can improve the performance of the SEDACS–LATTE interface on tailored HPC systems. Note that minor modifications to `spack_all.yaml` may be required to resolve dependency or compiler issues on different architectures and platforms.

### Setup SEDACS–LATTE Interface on NERSC Perlmutter

Clone SEDACS, LATTE, BML, and PROGRESS repos
```shell
git clone https://github.com/lanl/sedacs.git
git clone -b lattepy https://github.com/lanl/LATTE.git
git clone https://github.com/lanl/bml.git
git clone https://github.com/lanl/qmd-progress.git
```
Compile LATTE, BML, and PROGRESS libraries with Cray wrappers
```shell
git clone https://github.com/lanl/bml.git
source sedacs/envs/perlmutter/build_bml.sh
source sedacs/envs/perlmutter/build_progress.sh
cd LATTE/src
make
cd ..
```
Install Python dependencies
```
module load python
mamba create -n sedacs python=3.12 metis -c conda-forge --yes
mamba activate sedacs
pip install -r sedacs/envs/perlmutter/requirements.txt 
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```
Install SEDACS
```
cd sedacs
pip install -e .
```
Export env variables
```
export LATTE_PATH=~/LATTE/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/bml/install/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/bml/install/lib64/
```

---

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

---

# License O# (O4732)

This program is open source under the [BSD-3 License](LICENSE.txt).
