#!/usr/bin/env python3
"""Lightweight workload used to gather PGO profiles for interpn."""

from __future__ import annotations

import numpy as np

from interpn import MulticubicRectilinear, MulticubicRegular, MultilinearRectilinear, MultilinearRegular

_OBSERVATION_COUNTS = (1, 1000)
_MAX_DIMS = 8
_GRID_SIZE = 4


def _observation_points(
    rng: np.random.Generator, ndims: int, nobs: int, dtype: np.dtype
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate observation points inside the grid domain."""
    m = max(int(float(nobs) ** (1.0 / ndims) + 2.0), 2)
    axes = [np.linspace(-0.99, 0.99, m, dtype=dtype) for _ in range(ndims)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = [axis.flatten()[:nobs].copy() for axis in mesh]
    shuffled = [rng.permutation(axis) for axis in points]
    return points, shuffled


def _extrapolation_points(points: list[np.ndarray], dtype: np.dtype, offset: float) -> list[np.ndarray]:
    """Create extrapolation points by shifting coordinates outside the grid."""
    delta = np.asarray(offset, dtype=dtype)
    return [axis + delta for axis in points]


def _evaluate(interpolator, points: list[np.ndarray]) -> None:
    interpolator.eval(points)
    out = np.empty_like(points[0])
    interpolator.eval(points, out)


def main() -> None:
    rng = np.random.default_rng(0)

    for dtype in (np.float64, np.float32):
        for ndims in range(1, _MAX_DIMS + 1):
            grids = [np.linspace(-1.0, 1.0, _GRID_SIZE, dtype=dtype) for _ in range(ndims)]
            mesh = np.meshgrid(*grids, indexing="ij")
            zgrid = rng.uniform(-1.0, 1.0, mesh[0].size).astype(dtype)
            dims = [grid.size for grid in grids]
            starts = np.array([grid[0] for grid in grids], dtype=dtype)
            steps = np.array([grid[1] - grid[0] for grid in grids], dtype=dtype)

            linear_regular = MultilinearRegular.new(dims, starts, steps, zgrid)
            linear_rect = MultilinearRectilinear.new(grids, zgrid)
            cubic_regular = MulticubicRegular.new(
                dims,
                starts,
                steps,
                zgrid,
                linearize_extrapolation=(ndims % 2 == 0),
            )
            cubic_rect = MulticubicRectilinear.new(
                grids,
                zgrid,
                linearize_extrapolation=(ndims % 2 == 1),
            )

            for nobs in _OBSERVATION_COUNTS:
                base_points, shuffled_points = _observation_points(rng, ndims, nobs, dtype)
                outside_points = _extrapolation_points(base_points, dtype, offset=2.5)

                for points in (base_points, shuffled_points, outside_points):
                    for interpolator in (
                        linear_regular,
                        linear_rect,
                        cubic_regular,
                        cubic_rect,
                    ):
                        _evaluate(interpolator, points)

            print(f"Completed dtype={np.dtype(dtype).name} ndims={ndims}")


if __name__ == "__main__":
    main()
