# XL-BOMD with SEDACS–LATTE interface 

LATTE is an open source density functional tight binding molecular dynamics package. We built the SEDACS–LATTE interface on top of the `lattepy` branch in the [LATTE github repository](https://github.com/lanl/LATTE/tree/lattepy).

## Quick Start

Activate the Spack environment:
```shell
spack env activate -p sedacs 
```
With MPI:
```shell
mpirun -n 8 python main.py
```
Without MPI:
```shell
python main.py
```

## Command-line Options
```bash
python main.py --help
```
Outputs:
```text
usage: main.py [-h] [--device DEVICE] [--use-torch] [--input-file INPUT_FILE] [--md_iter MD_ITER] [--dt DT] [--temp TEMP] [--mu MU] [--localization LOCALIZATION] [--shadow_md SHADOW_MD]
               [--use_kernel USE_KERNEL]

Extended-Lagrangian Born-Oppenheimer molecular dynamics with SEDACS–LATTE interface

options:
  -h, --help            show this help message and exit
  --device DEVICE       CPU/GPU device
  --use-torch           Use pytorch
  --input-file INPUT_FILE
                        Specify input file
  --md_iter MD_ITER     Number of timesteps
  --dt DT               Timestep size (fs)
  --temp TEMP           Initial system temperature (K)
  --mu MU               Initial Chemical potential (eV)
  --localization LOCALIZATION
                        Degree of localization for adaptive halo expansion
  --shadow_md SHADOW_MD
                        Set to 1/0 to enable/disable shadow MD
  --use_kernel USE_KERNEL
                        Set to 1/0 to enable/disable kernel calculation
```
