"""
Re-exported raw PyO3/Maturin bindings to Rust functions.
Using these can yield some performance benefit at the expense of ergonomics.
"""

from ._interpn import (
    interpn_regular_f64,
    interpn_regular_f32,
    check_bounds_regular_f64,
    check_bounds_regular_f32,
    interpn_rectilinear_f64,
    interpn_rectilinear_f32,
    check_bounds_rectilinear_f64,
    check_bounds_rectilinear_f32,
)

__all__ = [
    "interpn_regular_f64",
    "interpn_regular_f32",
    "check_bounds_regular_f64",
    "check_bounds_regular_f32",
    "interpn_rectilinear_f64",
    "interpn_rectilinear_f32",
    "check_bounds_rectilinear_f64",
    "check_bounds_rectilinear_f32",
]
