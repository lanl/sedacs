# Installation Guide for sedacs

```{contents} Table of Contents
:depth: 3
```

This document provides comprehensive instructions on how to install the `sedacs` project. We offer several methods for installation, including using pip, Conda, and PDM (Python Development Master).

## Installing Using pip

pip is a popular tool that allows you to install and manage software packages written in Python.

### Install from PyPI

To install the latest version of `sedacs` directly from the Python Package Index (PyPI), use the following command:

````shell
pip install sedacs
````

### Install from Local Source in Editable Mode

If you have a local copy of the `sedacs` source code and want to install it in a way that allows for dynamic updates to the source code (editable mode), navigate to the directory containing `pyproject.toml` and run:

````shell
pip install -e .
````

## Installing Using Conda

Conda is an open-source package management and environment management system which is widely used in scientific computing and data science.

To install `sedacs` from Conda, use the following command:

````shell
conda install -c conda-forge sedacs
````

Ensure that you have the `conda-forge` channel added to your Conda configuration.

## Installing Using PDM

PDM (Python Development Master) is recommended for users who need robust dependency management and project management capabilities. It leverages PEP 582 to manage the Python path directly, avoiding the need to create virtual environments.

### Install from PyPI Using PDM

To install `sedacs` from PyPI using PDM, you can use the `add` command. This command adds the package to your `pyproject.toml` and installs it:

````shell
pdm add sedacs
````

### Install from Local Source Using PDM

To install `sedacs` from a local directory in editable mode, follow these steps:

1. Navigate to your project directory where `pyproject.toml` is located.

2. Install the package along with its dependencies:

   ````shell
   pdm install
   ````

## General PDM Commands

For general management of the project's environment and dependencies, PDM provides several useful commands, such as:

- `pdm build`: Build artifacts for distribution.
- `pdm update`: Update the packages as per the configuration in `pyproject.toml`.
- `pdm list`: List all packages installed in the current project environment.

Using PDM can significantly simplify Python package management, especially in complex projects.

## Conclusion

Depending on your needs and environment setup, you can choose any of the above methods to install `sedacs`. For most users, pip offers a quick and straightforward installation option. However, for those involved in development or managing multiple projects, PDM provides advanced tools and features that enhance the workflow.
