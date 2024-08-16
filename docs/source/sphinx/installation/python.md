# Python Installation Guide

```{contents} Table of Contents
:depth: 3
```

Python may already be installed on many operating systems. To check, execute the `python` command in the terminal to access the Python interpreter, or use the `py` command on Windows, which acts as a Python launcher. If Python is installed, the terminal will display its version, as shown below:

```python
Python 3.12.4 (main, Jul  4 2024, 23:12:25) [Clang 15.0.0 (clang-1500.3.9.4)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
```

If the terminal does not display a similar message, or if the version shown is Python 2.x.y, you will need to install or update to a newer version of Python, preferably Python 3.10 or later, as our package requires Python >= 3.10 to function correctly. Python 2 is outdated and not recommended for development.

To install or update Python, visit the [Python download page](https://www.python.org/downloads/).

This page will provide an installer suited to your OS. The Python documentation offers comprehensive installation guides at [Python's official documentation](https://docs.python.org/3/using/index.html).

## Installation Notes by Operating System

### Windows

For Windows, the stable version of Python can be downloaded from the [Python download page](https://www.python.org/downloads/).

After downloading, run the installer from the site. Detailed instructions on installation are available at [Windows installation guide](https://docs.python.org/3/using/windows.html).

### Mac

For macOS versions 10.9 to 12.3, Python 2 is pre-installed but is no longer supported. Users should visit the [Python download page](https://www.python.org/downloads/) to download and install a current version of Python. For newer macOS versions, Python must be installed manually as it is not included by default. Guidance on installation can be found at [Mac installation guide](https://docs.python.org/3/using/mac.html).

### Linux

Python is typically pre-installed or available through the package managers of most Linux distributions. For detailed, distribution-specific installation instructions, consult your distribution's documentation. To install Python from source or if your package manager does not provide the latest version, download the source tarball from the [Python download page](https://www.python.org/downloads/). Below are package manager commands for popular distributions:

#### Red Hat, CentOS, or Fedora

```sh
dnf install python3 python3-devel
```

#### Debian or Ubuntu

```sh
apt-get install python3 python3-dev
```

#### Gentoo

```sh
emerge dev-lang/python
```

#### Arch Linux

```sh
pacman -S python3
```

## Cross-platform Installation Methods

For a more consistent and versatile Python environment across different systems, we strongly recommend using cross-platform tools like pyenv, asdf, or PDM. These tools provide greater flexibility and control over Python installations and dependencies.

### pyenv and asdf

pyenv and asdf are preferred for managing multiple Python versions. They allow you to install and switch between versions effortlessly.

For example, to download and install the latest Python 3.12 release with pyenv, run:

```sh
pyenv install 3.12
pyenv global 3.12
```

With asdf, use the following commands:

```sh
asdf plugin-add python
asdf plugin update python
asdf install python 3.12.4
asdf global python 3.12.4
```

### PDM

PDM is a modern Python package and dependency manager that adheres to the latest PEP standards, making it an excellent choice for managing Python packages.

To use PDM for installing Python versions:

```sh
pdm py list
pdm py install
pdm py install 3.12
```

We have more detailed instructions on how to install PDM in [here](pdm.md).

### Conda

Conda is an open-source package management and environment management system supporting multiple languages, making it highly effective for scientific and data analysis projects.

To install Python with Conda, run:

```sh
conda create --name myenv python=3.10
conda activate myenv
```

This creates a new environment named `myenv` with Python 3.12 installed, which you can activate and use for your projects.

Remember to always update to the latest version of Python to ensure compatibility and security.
