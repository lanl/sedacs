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
perform up to a full SCF optimization of the DM. It is written in
three different languages: Python, Fortran, and C.

### `gpu`

This is an early implementation of the GPU/AI-solver library.

### `latte`

This is a code that generates "Latte" Hamiltonians from input coordinates
files (`xyz` or `pdb`) and constructs the density matrix.

### `mods`

Auxiliary Python modules. These modules will be used as building blocks
to develop SEDACS.

### `driver`

Scripts to exercise the code.


# License

This program is Open-Source under the BSD-3 License.
 
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
