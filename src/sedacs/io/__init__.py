import importlib.util
import os.path
from pathlib import Path

__all__ = ["src_path", "data_path", "plots_path", "examples_path"]


def locate_package(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec and spec.origin:
        # Get the path to the package
        package_path = spec.origin
        # Get the directory containing the package
        installation_path = os.path.dirname(package_path)
        return Path(installation_path)
    else:
        print(f"Package `{package_name}` not found.")


def src_path():
    return locate_package("sedacs")


def data_path():
    return os.path.join(src_path(), "data")


def plots_path():
    return os.path.join(src_path(), "plots")


def examples_path():
    return os.path.join(src_path(), "examples")
