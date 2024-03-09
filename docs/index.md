# Quickstart

[Repo (github)](https://github.com/jlogan03/interpnpy) | 
[Rust Docs (docs.rs)](https://docs.rs/interpn/latest/interpn/)

This library provides serializable N-dimensional interpolators
backed by compute-heavy code written in Rust.

These methods perform zero allocation when evaluated (except, optionally, for the output). 
Because of this, they have minimal per-call overhead, and are particularly
effective when examining small numbers of observation points. See the [performance](/perf) page for detailed benchmarks.

## Install

```bash
pip install interpn
```

## Features
| N-dimensional Grid Type →<br>↓ Interpolant Method | Regular | Rectilinear |
|-------------------|---------|-------------|
| Linear            |   ✅    |     ✅      |
| Cubic             |   ✅    |     💡      |


## Example: Multilinear Interpolation on a Regular Grid
```python
import interpn
import numpy as np

# Build grid
x = np.linspace(0.0, 10.0, 5)
y = np.linspace(20.0, 30.0, 3)

xgrid, ygrid = np.meshgrid(x, y, indexing="ij")
zgrid = (xgrid + 2.0 * ygrid)  # Values at grid points

# Grid inputs for true regular grid
dims = [x.size, y.size]
starts = np.array([x[0], y[0]])
steps = np.array([x[1] - x[0], y[1] - y[0]])

# Observation points pointed back at the grid
obs = [xgrid.flatten(), ygrid.flatten()]

# Initialize
interpolator = interpn.MultilinearRegular.new(
    dims, starts, steps, zgrid.flatten()
)

# Interpolate
out = interpolator.eval(obs)

# Check result
assert np.all(out == zgrid.flatten())

# Serialize and deserialize
roundtrip_interpolator = interpn.MultilinearRegular.model_validate_json(
    interpolator.model_dump_json()
)
out2 = roundtrip_interpolator.eval(obs)

# Check result from roundtrip serialized/deserialized interpolator
assert np.all(out == out2)
```
