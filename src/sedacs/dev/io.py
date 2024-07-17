import importlib.util
import os.path
from pathlib import Path

__all__ = ["project_path", "src_path", "bindeps_path", "data_path", "plots_path", "examples_path"]


def locate_package(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec and spec.origin:
        # Get the path to the package
        package_path = spec.origin
        # Get the directory containing the package
        installation_path = os.path.dirname(package_path)
        return Path(installation_path)
    raise ModuleNotFoundError(f"Package `{package_name}` not found.")


def src_path():
    return locate_package("sedacs")


def project_path():
    return src_path().parent.parent


def data_path():
    return project_path() / "data/"


def plots_path():
    return project_path() / "plots/"


def examples_path():
    return project_path() / "examples/"


def bindeps_path():
    return project_path() / "bindeps/"
