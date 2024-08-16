# PDM Installation Guide

```{contents} Table of Contents
:depth: 3
```

PDM (Python Development Master) is a modern Python package and dependency manager that supports the latest PEP standards, aimed at enhancing your development workflow.

## Prerequisites

- Python 3.8 or higher.

## Installation Methods

PDM can be installed via a script, package managers, or even directly through pip.

### Script Installation (Recommended)

This method installs PDM into an isolated environment.

#### Linux/macOS

```sh
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

#### Windows

```powershell
(Invoke-WebRequest -Uri https://pdm-project.org/install-pdm.py -UseBasicParsing).Content | py -
```

**Note:** If the `py` launcher is not available on Windows, use `python` instead.

### Package Managers

You can also install PDM using various package managers.

#### Homebrew (Linux/macOS)

```sh
brew install pdm
```

#### Scoop (Windows)

```powershell
scoop bucket add frostming https://github.com/frostming/scoop-frostming.git
scoop install pdm
```

#### uv (Cross-platform)

```sh
uv tool install pdm
```

#### pipx (Cross-platform)

```sh
pipx install pdm
# To install the head version from GitHub repository
pipx install git+https://github.com/pdm-project/pdm.git@main#egg=pdm
# To install PDM with all features
pipx install pdm[all]
```

#### asdf (Cross-platform)

```sh
asdf plugin add pdm
asdf local pdm latest
asdf install pdm
```

### Using pip

PDM can be installed via pip, suitable for all platforms.

```sh
pip install --user pdm
```

## Update PDM

Keep PDM up-to-date by running:

```sh
pdm self update
```

For more details, visit the official [PDM documentation](https://pdm-project.org/en/latest/#recommended-installation-method).
