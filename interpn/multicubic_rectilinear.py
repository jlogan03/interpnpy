from __future__ import annotations

from typing import Optional
from functools import reduce

import numpy as np
from numpy.typing import NDArray

from pydantic import (
    model_validator,
    ConfigDict,
    BaseModel,
)

from .serialization import Array, ArrayF32, ArrayF64

from ._interpn import (
    interpn_cubic_rectilinear_f64,
    interpn_cubic_rectilinear_f32,
    check_bounds_rectilinear_f64,
    check_bounds_rectilinear_f32,
)


class MulticubicRectilinear(BaseModel):
    """
    Multicubic interpolation on a rectilinear grid in up to 8 dimensions.

    This method uses a symmetrized Hermite spline interpolant,
    which provides a continuous value and first derivative.
    Unlike a B-spline, the second derivative is not continuous;
    however, also unlike a B-spline, the first derivatives are
    maintained to more exactly match the data as estimated by
    a central difference.

    If `linearize_extrapolation` is set, dimensions on which extrapolation is occurring
    (but not other dimensions) are extrapolated linearly from the last
    two grid points on that dimension.

    All array inputs must be of the same type, either np.float32 or np.float64
    and must be 1D and contiguous and have size at least 4.
    """

    # Immutable after initialization checks
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    grids: list[Array]
    vals: Array
    linearize_extrapolation: bool

    @classmethod
    def new(cls, grids: list[NDArray], vals: NDArray, linearize_extrapolation: bool = False) -> MulticubicRectilinear:
        """
        Initialize interpolator and check types and dimensions, casting other arrays
        to the same type as `vals` if they do not match, and flattening and/or
        reallocating into contiguous storage if necessary.

        This method exists primarily to remove boilerplate introduced by
        mixing pydantic and numpy.

        Args:
            grids: 1D arrays of grid coordinate values.
                   All grids must be monotonically increasing.
            vals: Values at grid points in C-style ordering,
                  as obtained from np.meshgrid(..., indexing="ij")
            linearize_extrapolation: Whether to fall back to a linear
                interpolant outside the grid

        Returns:
            A new MultilinearRectilinear interpolator instance.
        """
        dtype = vals.dtype
        arrtype = ArrayF64 if dtype == np.float64 else ArrayF32
        interpolator = MulticubicRectilinear(
            grids=[arrtype(data=x) for x in grids],
            vals=arrtype(data=vals.flatten()),
            linearize_extrapolation=linearize_extrapolation,
        )

        return interpolator

    @model_validator(mode="after")
    def _validate_model(self):
        """Check that all inputs are contiguous and of the same data type,
        and that the grid dimensions and values make sense."""
        dims = self.dims()
        ndims = self.ndims()
        assert (
            ndims <= 8 and ndims >= 1
        ), "Number of dimensions must be at least 1 and no more than 8"
        assert self.vals.data.size == reduce(
            lambda acc, x: acc * x, dims
        ), "Size of value array does not match grid dims"
        assert all(
            [np.all(np.diff(x.data) > 0.0) for x in self.grids]
        ), "All grids must be monotonically increasing"
        assert all(
            [x.data.dtype == self.vals.data.dtype for x in self.grids]
        ), "All grid inputs must be of the same data type (np.float32 or np.float64)"
        assert (
            all([x.data.data.contiguous for x in self.grids])
            and self.vals.data.data.contiguous
        ), "Grid data must be contiguous"

        return self

    def ndims(self) -> int:
        return len(self.grids)

    def dims(self) -> list[int]:
        return [x.data.size for x in self.grids]

    def eval(self, obs: list[NDArray], out: Optional[NDArray] = None) -> NDArray:
        """Evaluate the interpolator at a set of observation points,
        optionally writing the output into a preallocated array.

        This function does not reallocate inputs, and will error if the
        inputs are not contiguous or are of the wrong data type.

        Args:
            obs: [x, y, ...] coordinates of observation points.
            out: Optional preallocated array for output. Defaults to None.

        Raises:
            TypeError: If data type is not np.float32 or np.float64
            AssertionError: If input data is not contiguous or dimensions do not match

        Returns:
            Array of evaluated values in the same shape and data type as obs[0]
        """
        # Allocate output if it was not provided,
        # then check data type and contiguousness
        out_inner = out if out is not None else np.zeros_like(obs[0])
        self.eval_unchecked(obs, out_inner)

        return out_inner

    def eval_unchecked(
        self, obs: list[NDArray], out: Optional[NDArray] = None
    ) -> NDArray:
        """Evaluate the interpolator at a set of observation points,
        optionally writing the output into a preallocated array,
        and skipping checks on the dimensionality and contiguousness
        of the inputs.

        This function does not reallocate inputs, and will error in a lower-level
        function if the inputs are not contiguous or are of the wrong data type.

        Args:
            obs: [x, y, ...] coordinates of observation points.
            out: Optional preallocated array for output. Defaults to None.

        Raises:
            TypeError: If data type is not np.float32 or np.float64

        Returns:
            Array of evaluated values in the same shape and data type as obs[0]
        """
        dtype = self.vals.data.dtype
        out_inner = out if out is not None else np.zeros_like(obs[0])

        if dtype == np.float64:
            interpn_cubic_rectilinear_f64(
                [x.data for x in self.grids],
                self.vals.data,
                self.linearize_extrapolation,
                obs,
                out_inner,
            )
        elif dtype == np.float32:
            interpn_cubic_rectilinear_f32(
                [x.data for x in self.grids],
                self.vals.data,
                self.linearize_extrapolation,
                obs,
                out_inner,
            )
        else:
            raise TypeError(f"Unexpected data type: {dtype}")

        return out_inner

    def check_bounds(self, obs: list[NDArray], atol: float) -> NDArray[np.bool_]:
        """
        Check if the observation points violated the bounds on each dimension.

        This performs a (small) allocation for the output.

        Args:
            obs: [x, y, ...] coordinates of observation points.
            atol: Absolute tolerance on bounds.

        Raises:
            TypeError: If an unexpected data type is encountered

        Returns:
            An array of flags for each dimension, each True if that dimension's
            bounds were violated.
        """
        ndims = self.ndims()
        out = np.array([False] * ndims)

        dtype = self.vals.data.dtype
        if dtype == np.float64:
            check_bounds_rectilinear_f64(
                [x.data for x in self.grids],
                [x.flatten() for x in obs],
                atol,
                out,
            )
        elif dtype == np.float32:
            check_bounds_rectilinear_f32(
                [x.data for x in self.grids],
                [x.flatten() for x in obs],
                atol,
                out,
            )
        else:
            raise TypeError(f"Unexpected data type: {dtype}")

        return out
