from importlib.resources import files
from pathlib import Path

import os
import sys

global eng_path

fortran_proxy_path = "PROXYA_FORTRAN_PATH"
python_proxy_path = "PROXYA_PYTHON_PATH"

def init_eng_path(path):
    global eng_path
    eng_path = path

def init_proxies_path():
    proxies = files("sedacs").joinpath("../../proxies").resolve()
    _ensure_sys_path(str(files("sedacs").joinpath("../../").resolve()))
    _ensure_sys_path(str(files("sedacs").joinpath("./gpu/").resolve()))

    if fortran_proxy_path not in os.environ:
        os.environ[fortran_proxy_path] = str(proxies.joinpath("./fortran/build/").resolve())

    if python_proxy_path not in os.environ:
        os.environ[python_proxy_path] = str(proxies.joinpath("./python/").resolve()) 
        _ensure_sys_path(str(proxies.joinpath("./python/").resolve()))

    return proxies

def _ensure_sys_path(path_str, position=0):
    # Add absolute path once, at requested position
    abs_path = str(Path(path_str).resolve())
    # avoid duplicate entries (normalize)
    normalized = [str(Path(p).resolve()) for p in sys.path]
    if abs_path not in normalized:
        sys.path.insert(max(0, position), abs_path)
