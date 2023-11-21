"""
Python bindings to the `interpn` Rust library
for N-dimensional interpolation and extrapolation.
"""
from __future__ import annotations

from importlib.metadata import version

from .multilinear_regular import MultilinearRegular
from .multilinear_rectilinear import MultilinearRectilinear
from interpn import raw

__version__ = version("interpn")

__all__ = [
    "__version__",
    "MultilinearRegular",
    "MultilinearRectilinear",
    "raw",
]
