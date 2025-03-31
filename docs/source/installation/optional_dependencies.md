# Installation Guide for Optional Dependencies

```{contents} Table of Contents
:depth: 3
```

The following optional dependencies can be included as needed:

- **mdtraj**: For reading, writing, and analyzing molecular dynamics trajectories.
- **mpi**: Necessary for MPI support in distributed computing.
- **torch**: Integrates PyTorch for machine learning models.
- **graph**: Utilized for efficient graph partitioning and related tasks.

## Using PDM

This guide explains how to manage and install the optional dependencies of our project using PDM. Each dependency supports specific functionalities within our application, enhancing its modularity and allowing for a tailored setup.

You can selectively install these components using `pdm install -G <group>`. Here are the detailed instructions for managing these dependencies.

### Installing a Specific Optional Dependency

Install only the dependencies you need by specifying their respective groups:

```sh
pdm install -G mdtraj  # Installs only MDTraj
pdm install -G mpi     # Installs only mpi4py
pdm install -G torch   # Installs only PyTorch
pdm install -G graph   # Installs only METIS
```

### Installing Multiple Optional Dependencies

You can install multiple dependencies at once by listing their groups together:

```sh
pdm install -G mdtraj,torch  # Installs MDTraj and PyTorch
```

or

```sh
pdm install -G mdtraj -G torch
```

### Installing All Optional Dependencies

To install all available optional dependencies at once:

```sh
pdm install -G:all  # Installs all optional dependencies
```

### Tips and Additional Options

- **`--no-self`**: Use this if you do not want the root project to be installed.
- **`--no-editable`**: Applies if you want all packages installed in non-editable versions.

### Locking Dependencies

While the above commands install the dependencies directly, you may also lock them first, which helps in ensuring that subsequent installations are consistent:

```sh
pdm lock -G mdtraj  # Locks only MDTraj
pdm lock -G:all     # Locks all optional dependencies
pdm sync
```

## Using pip

When developing or testing locally, you may need to install your project along with specific optional dependencies defined in your `pyproject.toml`. This section provides detailed instructions on how to install these dependencies using pip, either locally or from PyPI.

### Defining Optional Dependencies

First, ensure your optional dependencies are correctly defined in your `pyproject.toml` under `[project.optional-dependencies]`. Here’s an example setup:

```toml
[project.optional-dependencies]
mdtraj = ["mdtraj"]
mpi = ["mpi4py"]
torch = ["torch"]
graph = ["metis"]
```

### Installing a Specific Optional Dependency

To install the project along with a specific optional dependency from PyPI or locally, use the following command:

```sh
pip install '.[mdtraj]'       # Installs the project from source with MDTraj
pip install 'sedacs[mdtraj]'  # Installs the project with MDTraj
```

This command installs the main project along with the `mdtraj` package. Replace `mdtraj` with `mpi`, `torch`, or `graph` as needed depending on which optional dependency you want to install.

### Installing Multiple Optional Dependencies

If your work requires multiple optional dependencies simultaneously, specify each group within the brackets separated by commas:

```sh
pip install 'sedacs[mdtraj,mpi]'  # Installs the project with MDTraj and mpi4py
```

### Installing All Optional Dependencies

To install all defined optional dependencies at once, you can use:

```sh
pip install 'sedacs[mdtraj,mpi,torch,graph]'  # Installs all optional dependencies
```

## Conclusion

Both PDM and pip offer flexible installation strategies that allow you to configure your development environment with only the necessary components, reducing setup complexity and optimizing resource use. Using `pdm install -G` and `pip install 'package[extras]'` ensures that your setup can be easily tailored, replicated, or modified to meet the specific needs of development and testing environments. This approach is particularly useful for managing various configurations of optional dependencies in real-time, enhancing both the modularity and functionality of the project.
