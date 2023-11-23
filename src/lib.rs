use numpy::PyArray1;
use pyo3::exceptions;
use pyo3::prelude::*;

use interpn::multilinear::rectilinear;
use interpn::multilinear::regular;

macro_rules! interpn_regular_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname(
            dims: Vec<usize>, // numpy index arrays are signed; this avoids casting
            starts: &PyArray1<$T>,
            steps: &PyArray1<$T>,
            vals: &PyArray1<$T>,
            obs: Vec<&PyArray1<$T>>,
            out: &PyArray1<$T>,
        ) -> PyResult<()> {
            // Unpack inputs
            let startsro = starts.readonly();
            let starts = startsro.as_slice()?;

            let stepsro = steps.readonly();
            let steps = stepsro.as_slice()?;

            let valsro = vals.readonly();
            let vals = valsro.as_slice()?;

            let obsro: Vec<_> = obs.iter().map(|&x| x.readonly()).collect();
            let mut obsvec: Vec<_> = Vec::with_capacity(obs.len());
            obsro.iter().try_for_each(|x| {
                let res = x.as_slice();
                match res {
                    Ok(xslice) => {
                        obsvec.push(xslice);
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            })?;
            let obs = &obsvec[..];

            // Get output as mutable
            let mut outrw = out.try_readwrite()?;
            let out = outrw.as_slice_mut()?;

            // Check lengths
            let ndims = dims.len();
            let grid_lengths_good =
                starts.len() == ndims && steps.len() == ndims && obs.len() == ndims;
            let vals_length_good = vals.len() == dims.iter().product();
            let obs_lengths_good = obs.iter().all(|&x| x.len() == out.len());
            if !grid_lengths_good {
                return Err(exceptions::PyAssertionError::new_err(
                    "Grid input dimensions do not match.",
                ));
            }
            if !vals_length_good {
                return Err(exceptions::PyAssertionError::new_err(
                    "Length of grid values does not match grid dimensions.",
                ));
            }
            if !obs_lengths_good {
                return Err(exceptions::PyAssertionError::new_err(
                    "Observation point coordinate lengths do not match output length.",
                ));
            }

            // Check for zero-size steps and degenerate dimensions
            if steps.iter().any(|&x| x == 0.0) {
                return Err(exceptions::PyAssertionError::new_err(
                    "All step sizes must be nonzero.",
                ));
            }
            if dims.iter().any(|&x| x < 2) {
                return Err(exceptions::PyAssertionError::new_err(
                    "All dimensions must have size at least 2.",
                ));
            }

            // Evaluate
            regular::interpn(&dims, starts, steps, vals, obs, out);

            Ok(())
        }
    };
}

interpn_regular_impl!(interpn_regular_f64, f64);
interpn_regular_impl!(interpn_regular_f32, f32);

macro_rules! check_bounds_regular_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname(
            dims: Vec<usize>, // numpy index arrays are signed; this avoids casting
            starts: &PyArray1<$T>,
            steps: &PyArray1<$T>,
            obs: Vec<&PyArray1<$T>>,
            atol: $T,
            out: &PyArray1<bool>,
        ) -> PyResult<()> {
            // Unpack inputs
            let startsro = starts.readonly();
            let starts = startsro.as_slice()?;

            let stepsro = steps.readonly();
            let steps = stepsro.as_slice()?;

            let obsro: Vec<_> = obs.iter().map(|&x| x.readonly()).collect();
            let mut obsvec: Vec<_> = Vec::with_capacity(obs.len());
            obsro.iter().try_for_each(|x| {
                let res = x.as_slice();
                match res {
                    Ok(xslice) => {
                        obsvec.push(xslice);
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            })?;
            let obs = &obsvec[..];

            // Get output as mutable
            let mut outrw = out.try_readwrite()?;
            let out = outrw.as_slice_mut()?;

            // Check lengths
            let ndims = dims.len();
            let grid_lengths_good =
                starts.len() == ndims && steps.len() == ndims && obs.len() == ndims;
            let obs_lengths_good = obs.iter().all(|&x| x.len() == obs[0].len());
            if !grid_lengths_good {
                return Err(exceptions::PyAssertionError::new_err(
                    "Grid input dimensions do not match.",
                ));
            }
            if !obs_lengths_good {
                return Err(exceptions::PyAssertionError::new_err(
                    "Observation point coordinate lengths do not match.",
                ));
            }

            // Check for zero-size steps and degenerate dimensions
            if steps.iter().any(|&x| x == 0.0) {
                return Err(exceptions::PyAssertionError::new_err(
                    "All step sizes must be nonzero.",
                ));
            }
            if dims.iter().any(|&x| x < 2) {
                return Err(exceptions::PyAssertionError::new_err(
                    "All dimensions must have size at least 2.",
                ));
            }

            // Evaluate
            regular::check_bounds(&dims, starts, steps, obs, atol, out);

            Ok(())
        }
    };
}

check_bounds_regular_impl!(check_bounds_regular_f64, f64);
check_bounds_regular_impl!(check_bounds_regular_f32, f32);

macro_rules! interpn_rectilinear_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname(
            grids: Vec<&PyArray1<$T>>,
            vals: &PyArray1<$T>,
            obs: Vec<&PyArray1<$T>>,
            out: &PyArray1<$T>,
        ) -> PyResult<()> {
            // Unpack inputs
            let gridsro: Vec<_> = grids.iter().map(|&x| x.readonly()).collect();
            let mut gridsvec: Vec<_> = Vec::with_capacity(obs.len());
            gridsro.iter().try_for_each(|x| {
                let res = x.as_slice();
                match res {
                    Ok(xslice) => {
                        gridsvec.push(xslice);
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            })?;
            let grids = &gridsvec[..];

            let valsro = vals.readonly();
            let vals = valsro.as_slice()?;

            let obsro: Vec<_> = obs.iter().map(|&x| x.readonly()).collect();
            let mut obsvec: Vec<_> = Vec::with_capacity(obs.len());
            obsro.iter().try_for_each(|x| {
                let res = x.as_slice();
                match res {
                    Ok(xslice) => {
                        obsvec.push(xslice);
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            })?;
            let obs = &obsvec[..];

            // Get output as mutable
            let mut outrw = out.try_readwrite()?;
            let out = outrw.as_slice_mut()?;

            // Check lengths
            let dims: Vec<usize> = grids.iter().map(|x| x.len()).collect();

            let vals_length_good = vals.len() == dims.iter().product();
            let obs_lengths_good = obs.iter().all(|&x| x.len() == out.len());
            if !vals_length_good {
                return Err(exceptions::PyAssertionError::new_err(
                    "Length of grid values does not match grid dimensions.",
                ));
            }
            if !obs_lengths_good {
                return Err(exceptions::PyAssertionError::new_err(
                    "Observation point coordinate lengths do not match output length.",
                ));
            }

            // Check for zero-size steps and degenerate dimensions
            if dims.iter().any(|&x| x < 2) {
                return Err(exceptions::PyAssertionError::new_err(
                    "All dimensions must have size at least 2.",
                ));
            }
            if grids.iter().any(|&x| (x[1] - x[0]) <= 0.0) {
                return Err(exceptions::PyAssertionError::new_err(
                    "All grid step sizes must be nonzero (grids must be monotonically increasing).",
                ));
            }

            // Evaluate
            rectilinear::interpn(grids, vals, obs, out);

            Ok(())
        }
    };
}

interpn_rectilinear_impl!(interpn_rectilinear_f64, f64);
interpn_rectilinear_impl!(interpn_rectilinear_f32, f32);

macro_rules! check_bounds_rectilinear_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname(
            grids: Vec<&PyArray1<$T>>,
            obs: Vec<&PyArray1<$T>>,
            atol: $T,
            out: &PyArray1<bool>,
        ) -> PyResult<()> {
            // Unpack inputs
            let gridsro: Vec<_> = grids.iter().map(|&x| x.readonly()).collect();
            let mut gridsvec: Vec<_> = Vec::with_capacity(obs.len());
            gridsro.iter().try_for_each(|x| {
                let res = x.as_slice();
                match res {
                    Ok(xslice) => {
                        gridsvec.push(xslice);
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            })?;
            let grids = &gridsvec[..];

            let obsro: Vec<_> = obs.iter().map(|&x| x.readonly()).collect();
            let mut obsvec: Vec<_> = Vec::with_capacity(obs.len());
            obsro.iter().try_for_each(|x| {
                let res = x.as_slice();
                match res {
                    Ok(xslice) => {
                        obsvec.push(xslice);
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            })?;
            let obs = &obsvec[..];

            // Get output as mutable
            let mut outrw = out.try_readwrite()?;
            let out = outrw.as_slice_mut()?;

            // Check lengths
            let dims: Vec<usize> = grids.iter().map(|x| x.len()).collect();
            let obs_lengths_good = obs.iter().all(|&x| x.len() == obs[0].len());
            if !obs_lengths_good {
                return Err(exceptions::PyAssertionError::new_err(
                    "Observation point coordinate lengths do not match output length.",
                ));
            }

            // Check for zero-size steps and degenerate dimensions
            if dims.iter().any(|&x| x < 2) {
                return Err(exceptions::PyAssertionError::new_err(
                    "All dimensions must have size at least 2.",
                ));
            }
            if grids.iter().any(|&x| (x[1] - x[0]) <= 0.0) {
                return Err(exceptions::PyAssertionError::new_err(
                    "All grid step sizes must be nonzero (grids must be monotonically increasing).",
                ));
            }

            // Evaluate
            rectilinear::check_bounds(&grids, obs, atol, out);

            Ok(())
        }
    };
}

check_bounds_rectilinear_impl!(check_bounds_rectilinear_f64, f64);
check_bounds_rectilinear_impl!(check_bounds_rectilinear_f32, f32);

/// Python bindings for select functions from `interpn`.
#[pymodule]
#[pyo3(name = "_interpn")]
fn interpnpy(_py: Python, m: &PyModule) -> PyResult<()> {
    // Multilinear regular grid
    m.add_function(wrap_pyfunction!(interpn_regular_f64, m)?)?;
    m.add_function(wrap_pyfunction!(interpn_regular_f32, m)?)?;
    m.add_function(wrap_pyfunction!(check_bounds_regular_f64, m)?)?;
    m.add_function(wrap_pyfunction!(check_bounds_regular_f32, m)?)?;
    // Multilinear rectilinear grid
    m.add_function(wrap_pyfunction!(interpn_rectilinear_f64, m)?)?;
    m.add_function(wrap_pyfunction!(interpn_rectilinear_f32, m)?)?;
    m.add_function(wrap_pyfunction!(check_bounds_rectilinear_f64, m)?)?;
    m.add_function(wrap_pyfunction!(check_bounds_rectilinear_f32, m)?)?;
    Ok(())
}
