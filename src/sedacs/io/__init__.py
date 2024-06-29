import importlib.util
import os.path

__all__ = ["packagepath", "datapath", "plotspath"]


def locate_package(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec and spec.origin:
        # Get the path to the package
        package_path = spec.origin
        # Get the directory containing the package
        installation_path = os.path.dirname(package_path)
        return installation_path
    else:
        print(f"Package `{package_name}` not found.")


def packagepath():
    return locate_package("sedacs")


def datapath():
    return os.path.join(packagepath(), "data")


def plotspath():
    return os.path.join(packagepath(), "plots")
