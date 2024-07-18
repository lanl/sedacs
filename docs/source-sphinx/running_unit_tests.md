# Running Unit Tests

This document provides instructions on setting up and running unit tests for our project using `pytest` and `pytest-cov`, as well as using the `pdm` tool.
The *root directory* of the project is the highest level directory that includes the `pyproject.toml` file.

## Prerequisites

Before running the tests, ensure that Python is installed on your system. It is recommended to use a virtual environment to manage dependencies.

### Installation with pip

Install `pytest` and `pytest-cov` using pip with the following command:

```bash
pip install -U pytest pytest-cov
```

This command installs `pytest`, our testing framework, and `pytest-cov`, a plugin for measuring code coverage.

### Installation with Conda

If you are using Conda, you can install `pytest` and `pytest-cov` with the following commands:

```bash
conda install -c anaconda pytest
conda install -c conda-forge pytest-cov
```

These commands install `pytest` and `pytest-cov` from the Anaconda and conda-forge repositories, respectively.

## Running Tests with pytest

Navigate to the root directory of the project and execute:

```bash
pytest
```

This command will discover and run all tests located in the `tests` directory.

## Running Tests with pdm

If you are using `pdm` as your package manager and build tool, you can run the tests through `pdm` to ensure all dependencies are managed properly:

```bash
pdm run pytest
```

This will execute `pytest` using the Python interpreter and dependencies managed by `pdm`.

## Running Tests in VSCode

To run tests within Visual Studio Code (VSCode), follow these steps:

1. Install the Python extension for VSCode, if not already installed.
2. Open the root directory of your project in VSCode.
3. Open the Command Palette (`Ctrl+Shift+P`) and select "Python: Configure Tests".
4. Choose `pytest` as the testing framework when prompted.
5. Once tests are discovered, you can run them using the Test Explorer in VSCode or by right-clicking on a test file and selecting "Run All Tests".

For more information, see [Python testing in Visual Studio Code](https://code.visualstudio.com/docs/python/testing).

## Additional Information

For more advanced features such as generating HTML reports or integrating with a CI/CD pipeline, refer to the official `pytest` documentation at [pytest documentation](https://docs.pytest.org/en/stable/).
