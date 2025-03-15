use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions;
use pyo3::prelude::*;

use interpn::multicubic;
use interpn::multilinear;

/// Maximum number of dimensions for linear interpn convenience methods
const MAXDIMS: usize = 8;

macro_rules! unpack_vec_of_arr {
    ($inname:ident, $outname:ident, $T:ty) => {
        // We need a mutable slice-of-slice,
        // and it has to start with a reference to something
        let dummy = [0.0; 0];
        let mut _arr: [&[$T]; MAXDIMS] = [&dummy[..]; MAXDIMS];
        // PyArray readonly references are very lightweight
        // but aren't Copy, so we can't template them out like
        // [...; 8]
        let mut _ro: [_; 8] = core::array::from_fn(|_| None);
        let n = $inname.len();
        (0..n).for_each(|i| _ro[i] = Some($inname[i].readonly()));
        for i in 0..n {
            match (&_ro[i]).as_ref() {
                Some(thisro) => _arr[i] = &thisro.as_slice()?,
                None => {
                    return Err(exceptions::PyAssertionError::new_err(
                        "Failed to unpack input array",
                    ))
                }
            }
        }
        let $outname = &_arr[..n];
    };
}

macro_rules! interpn_linear_regular_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname<'py>(
            dims: Vec<usize>, // numpy index arrays are signed; this avoids casting
            starts: Bound<'py, PyArray1<$T>>,
            steps: Bound<'py, PyArray1<$T>>,
            vals: Bound<'py, PyArray1<$T>>,
            obs: Vec<Bound<'py, PyArray1<$T>>>,
            out: Bound<'py, PyArray1<$T>>,
        ) -> PyResult<()> {
            // Unpack inputs
            let startsro = starts.readonly();
            let starts = startsro.as_slice()?;

            let stepsro = steps.readonly();
            let steps = stepsro.as_slice()?;

            let valsro = vals.readonly();
            let vals = valsro.as_slice()?;

            unpack_vec_of_arr!(obs, obs, $T);

            // Get output as mutable
            let mut outrw = out.try_readwrite()?;
            let out = outrw.as_slice_mut()?;

            // Evaluate
            match multilinear::regular::interpn(&dims, starts, steps, vals, obs, out) {
                Ok(()) => Ok(()),
                Err(msg) => Err(exceptions::PyAssertionError::new_err(msg)),
            }
        }
    };
}

interpn_linear_regular_impl!(interpn_linear_regular_f64, f64);
interpn_linear_regular_impl!(interpn_linear_regular_f32, f32);

macro_rules! check_bounds_regular_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname<'py>(
            dims: Vec<usize>, // numpy index arrays are signed; this avoids casting
            starts: Bound<'py, PyArray1<$T>>,
            steps: Bound<'py, PyArray1<$T>>,
            obs: Vec<Bound<'py, PyArray1<$T>>>,
            atol: $T,
            out: Bound<'py, PyArray1<bool>>,
        ) -> PyResult<()> {
            // Unpack inputs
            let startsro = starts.readonly();
            let starts = startsro.as_slice()?;

            let stepsro = steps.readonly();
            let steps = stepsro.as_slice()?;

            unpack_vec_of_arr!(obs, obs, $T);

            // Get output as mutable
            let mut outrw = out.try_readwrite()?;
            let out = outrw.as_slice_mut()?;

            // Evaluate
            match multilinear::regular::check_bounds(&dims, starts, steps, obs, atol, out) {
                Ok(()) => Ok(()),
                Err(msg) => Err(exceptions::PyAssertionError::new_err(msg)),
            }
        }
    };
}

check_bounds_regular_impl!(check_bounds_regular_f64, f64);
check_bounds_regular_impl!(check_bounds_regular_f32, f32);

macro_rules! interpn_linear_rectilinear_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname<'py>(
            grids: Vec<Bound<'py, PyArray1<$T>>>,
            vals: Bound<'py, PyArray1<$T>>,
            obs: Vec<Bound<'py, PyArray1<$T>>>,
            out: Bound<'py, PyArray1<$T>>,
        ) -> PyResult<()> {
            // Unpack inputs
            unpack_vec_of_arr!(grids, grids, $T);

            let valsro = vals.readonly();
            let vals = valsro.as_slice()?;

            unpack_vec_of_arr!(obs, obs, $T);

            // Get output as mutable
            let mut outrw = out.try_readwrite()?;
            let out = outrw.as_slice_mut()?;

            // Evaluate
            match multilinear::rectilinear::interpn(grids, vals, obs, out) {
                Ok(()) => Ok(()),
                Err(msg) => Err(exceptions::PyAssertionError::new_err(msg)),
            }
        }
    };
}

interpn_linear_rectilinear_impl!(interpn_linear_rectilinear_f64, f64);
interpn_linear_rectilinear_impl!(interpn_linear_rectilinear_f32, f32);

macro_rules! check_bounds_rectilinear_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname<'py>(
            grids: Vec<Bound<'py, PyArray1<$T>>>,
            obs: Vec<Bound<'py, PyArray1<$T>>>,
            atol: $T,
            out: Bound<'py, PyArray1<bool>>,
        ) -> PyResult<()> {
            // Unpack inputs
            unpack_vec_of_arr!(grids, grids, $T);
            unpack_vec_of_arr!(obs, obs, $T);

            // Get output as mutable
            let mut outrw = out.try_readwrite()?;
            let out = outrw.as_slice_mut()?;

            // Evaluate
            match multilinear::rectilinear::check_bounds(&grids, obs, atol, out) {
                Ok(()) => Ok(()),
                Err(msg) => Err(exceptions::PyAssertionError::new_err(msg)),
            }
        }
    };
}

check_bounds_rectilinear_impl!(check_bounds_rectilinear_f64, f64);
check_bounds_rectilinear_impl!(check_bounds_rectilinear_f32, f32);

macro_rules! interpn_cubic_regular_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname<'py>(
            dims: Vec<usize>, // numpy index arrays are signed; this avoids casting
            starts: Bound<'py, PyArray1<$T>>,
            steps: Bound<'py, PyArray1<$T>>,
            vals: Bound<'py, PyArray1<$T>>,
            linearize_extrapolation: bool,
            obs: Vec<Bound<'py, PyArray1<$T>>>,
            out: Bound<'py, PyArray1<$T>>,
        ) -> PyResult<()> {
            // Unpack inputs
            let startsro = starts.readonly();
            let starts = startsro.as_slice()?;

            let stepsro = steps.readonly();
            let steps = stepsro.as_slice()?;

            let valsro = vals.readonly();
            let vals = valsro.as_slice()?;

            unpack_vec_of_arr!(obs, obs, $T);

            // Get output as mutable
            let mut outrw = out.try_readwrite()?;
            let out = outrw.as_slice_mut()?;

            // Evaluate
            match multicubic::regular::interpn(
                &dims,
                starts,
                steps,
                vals,
                linearize_extrapolation,
                obs,
                out,
            ) {
                Ok(()) => Ok(()),
                Err(msg) => Err(exceptions::PyAssertionError::new_err(msg)),
            }
        }
    };
}

interpn_cubic_regular_impl!(interpn_cubic_regular_f64, f64);
interpn_cubic_regular_impl!(interpn_cubic_regular_f32, f32);

macro_rules! interpn_cubic_rectilinear_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname<'py>(
            grids: Vec<Bound<'py, PyArray1<$T>>>,
            vals: Bound<'py, PyArray1<$T>>,
            linearize_extrapolation: bool,
            obs: Vec<Bound<'py, PyArray1<$T>>>,
            out: Bound<'py, PyArray1<$T>>,
        ) -> PyResult<()> {
            // Unpack inputs
            unpack_vec_of_arr!(grids, grids, $T);

            let valsro = vals.readonly();
            let vals = valsro.as_slice()?;

            unpack_vec_of_arr!(obs, obs, $T);

            // Get output as mutable
            let mut outrw = out.try_readwrite()?;
            let out = outrw.as_slice_mut()?;

            // Evaluate
            match multicubic::rectilinear::interpn(grids, vals, linearize_extrapolation, obs, out) {
                Ok(()) => Ok(()),
                Err(msg) => Err(exceptions::PyAssertionError::new_err(msg)),
            }
        }
    };
}

interpn_cubic_rectilinear_impl!(interpn_cubic_rectilinear_f64, f64);
interpn_cubic_rectilinear_impl!(interpn_cubic_rectilinear_f32, f32);

/// Python bindings for select functions from `interpn`.
#[pymodule]
#[pyo3(name = "_interpn")]
fn interpnpy<'py>(_py: Python, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // Multilinear regular grid
    m.add_function(wrap_pyfunction!(interpn_linear_regular_f64, m)?)?;
    m.add_function(wrap_pyfunction!(interpn_linear_regular_f32, m)?)?;
    m.add_function(wrap_pyfunction!(check_bounds_regular_f64, m)?)?;
    m.add_function(wrap_pyfunction!(check_bounds_regular_f32, m)?)?;
    // Multilinear rectilinear grid
    m.add_function(wrap_pyfunction!(interpn_linear_rectilinear_f64, m)?)?;
    m.add_function(wrap_pyfunction!(interpn_linear_rectilinear_f32, m)?)?;
    m.add_function(wrap_pyfunction!(check_bounds_rectilinear_f64, m)?)?;
    m.add_function(wrap_pyfunction!(check_bounds_rectilinear_f32, m)?)?;
    // Multicubic with regular grid
    m.add_function(wrap_pyfunction!(interpn_cubic_regular_f64, m)?)?;
    m.add_function(wrap_pyfunction!(interpn_cubic_regular_f32, m)?)?;
    // Multicubic with rectilinear grid
    m.add_function(wrap_pyfunction!(interpn_cubic_rectilinear_f64, m)?)?;
    m.add_function(wrap_pyfunction!(interpn_cubic_rectilinear_f32, m)?)?;
    Ok(())
}
