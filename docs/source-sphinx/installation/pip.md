# Installing Packages with pip

```{contents} Table of Contents
:depth: 3
```

## Introduction

This guide explains how to install Python packages, which are bundles of software that can be installed using pip, a package management system. The term "package" here refers to what is technically known as a distribution.

## Pre-requisites

### Check Python Installation

Ensure Python is installed and can be accessed from the command line:

- **Unix/macOS**: `python3 --version`
- **Windows**: `py --version`

If Python is not installed, download it from [python.org](https://www.python.org) or refer to the [Hitchhiker's Guide to Python](https://docs.python-guide.org/starting/installation/).

### Verify pip Availability

Ensure `pip` is available on your system:

- **Unix/macOS**: `python3 -m pip --version`
- **Windows**: `py -m pip --version`

If `pip` is not installed, you can use:

- **Unix/macOS**: `python3 -m ensurepip --default-pip`
- **Windows**: `py -m ensurepip --default-pip`

Or download and run [get-pip.py](https://bootstrap.pypa.io/get-pip.py).

## Installing Packages

To install packages using `pip`, use the following commands:

- **Install a package**: `pip install SomeProject`
- **Install a specific version**: `pip install SomeProject==1.4`
- **Upgrade a package**: `pip install --upgrade SomeProject`

## Virtual Environments

Virtual environments allow you to manage separate package installations for different projects. To create and activate a virtual environment:

- **Unix/macOS**:
  ```bash
  python3 -m venv myenv
  source myenv/bin/activate
  ```
- **Windows**:
  ```bat
  py -m venv myenv
  myenv\Scripts\activate
  ```
