## Installation Guide for Optional Dependencies Using PDM

This guide explains how to manage and install the optional dependencies of our project using PDM. Each dependency supports specific functionalities within our application, enhancing its modularity and allowing for a tailored setup.

### Overview of Optional Dependencies

The following optional dependencies can be included as needed:

- **mdtraj**: For reading, writing, and analyzing molecular dynamics trajectories.
- **mpi**: Necessary for MPI support in distributed computing.
- **torch**: Integrates PyTorch for machine learning models.
- **metis**: Utilized for efficient graph partitioning and related tasks.

### Installing Optional Dependencies

You can selectively install these components using `pdm install -G <group>`. Here are the detailed instructions for managing these dependencies:

#### Individual Installation

Install only the dependencies you need by specifying their respective groups:

```bash
pdm install -G mdtraj  # Installs only mdtraj
pdm install -G mpi     # Installs only mpi
pdm install -G torch   # Installs only torch
pdm install -G metis   # Installs only metis
```

#### Grouped Installation

You can install multiple dependencies at once by listing their groups together:

```bash
pdm install -G mdtraj,torch  # Installs mdtraj and torch
```

or

```bash
pdm install -G mdtraj -G torch
```

#### Installing All Optional Dependencies

To install all available optional dependencies at once:

```bash
pdm install -G:all  # Installs all optional dependencies
```

### Tips and Additional Options

- **`--no-self`**: Use this if you do not want the root project to be installed.
- **`--no-editable`**: Applies if you want all packages installed in non-editable versions.

### Locking Dependencies

While the above commands install the dependencies directly, you may also lock them first, which helps in ensuring that subsequent installations are consistent:

```bash
pdm lock -G mdtraj  # Locks only mdtraj
pdm lock -G:all     # Locks all optional dependencies
pdm sync
```

## Conclusion

Using `pdm install -G` allows for a flexible installation strategy, letting you configure your development environment with only the necessary components, reducing setup complexity and optimizing resource use. For comprehensive management, ensure you use the correct group names as listed in your `pyproject.toml`.
