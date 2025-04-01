import ctypes
import pathlib

__all__ = ["Library"]


class Library:
    def __init__(self, path) -> None:
        self._path = pathlib.Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"library {path} not found.")

    @property
    def path(self) -> pathlib.Path:
        return self._path.absolute()

    def as_dll(self):
        return ctypes.CDLL(self.path.absolute().as_posix())

    def __repr__(self):
        return f'{type(self).__name__}("{self.path}")'

    def __str__(self) -> str:
        return str(self.__repr__())
