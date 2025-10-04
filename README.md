# interpn

Python bindings to the `interpn` Rust library for N-dimensional interpolation and extrapolation. 

[Docs](https://interpnpy.readthedocs.io/en/latest/) |
[Repo](https://github.com/jlogan03/interpnpy) |
[Rust Library (github)](https://github.com/jlogan03/interpn) | 
[Rust Docs (docs.rs)](https://docs.rs/interpn/latest/interpn/)

## Features

| Feature →<br>↓ Interpolant Method | Regular<br>Grid | Rectilinear<br>Grid | Json<br>Serialization |
|-----------------------------------|-----------------|---------------------|-----------------------|
| Linear                            |   ✅            |     ✅              | ✅                    |
| Cubic                             |   ✅            |     ✅              | ✅                    |

The methods provided here, while more limited in scope than scipy's,

* are significantly faster under most conditions
* use almost no RAM (and perform no heap allocations at all)
* produce significantly improved floating-point error (by several orders of magnitude)
* are json-serializable using Pydantic
* can also be used easily in web and embedded applications via the Rust library
* are permissively licensed

![ND throughput 1000 obs](./docs/throughput_vs_dims_1000_obs.svg)

See [here](https://interpnpy.readthedocs.io/en/latest/perf/) for more info about quality-of-fit, throughput, and memory usage.

## Installation

```bash
pip install interpn
```

## Profile-Guided Optimisation

To build the extension with profile-guided optimisation, using the lightweight `scripts/profile_workload.py` workload (1 and 1000 observation points across 1–8 dimensions for every InterpN method) by default:

1. Ensure the cargo subcommand is installed: `cargo install cargo-pgo`
2. Install the optional benchmarking dependencies if you plan to run the full SciPy benchmarks: `uv pip install '.[bench]'`
3. Run the automation script: `python scripts/run_pgo.py`

The helper uses `cargo-pgo` to build an instrumented extension, executes `scripts/profile_workload.py` to generate LLVM `.profraw` files, and then rebuilds the module with the merged profile data before copying the optimised library into `interpn/_interpn*.so`. Pass `--bench test/bench_cpu.py` to reuse the original comprehensive workload. Using `--skip-final-build` leaves the instrumented library in place alongside the collected profiles.

## Example: Available Methods

```python
import interpn
import numpy as np

# Build grid
x = np.linspace(0.0, 10.0, 5)
y = np.linspace(20.0, 30.0, 4)
grids = [x, y]

xgrid, ygrid = np.meshgrid(x, y, indexing="ij")
zgrid = (xgrid + 2.0 * ygrid)  # Values at grid points

# Grid inputs for true regular grid
dims = [x.size, y.size]
starts = np.array([x[0], y[0]])
steps = np.array([x[1] - x[0], y[1] - y[0]])

# Initialize different interpolators
# Call like `linear_regular.eval([xs, ys])`
linear_regular = interpn.MultilinearRegular.new(dims, starts, steps, zgrid)
cubic_regular = interpn.MulticubicRegular.new(dims, starts, steps, zgrid)
linear_rectilinear = interpn.MultilinearRectilinear.new(grids, zgrid)
cubic_rectilinear = interpn.MulticubicRectilinear.new(grids, zgrid)
```

## Example: Multilinear Interpolation on a Regular Grid

```python
import interpn
import numpy as np

# Build grid
x = np.linspace(0.0, 10.0, 5)
y = np.linspace(20.0, 30.0, 4)

xgrid, ygrid = np.meshgrid(x, y, indexing="ij")
zgrid = (xgrid + 2.0 * ygrid)  # Values at grid points

# Grid inputs for true regular grid
dims = [x.size, y.size]
starts = np.array([x[0], y[0]])
steps = np.array([x[1] - x[0], y[1] - y[0]])

# Observation points pointed back at the grid
obs = [xgrid.flatten(), ygrid.flatten()]

# Initialize
interpolator = interpn.MultilinearRegular.new(dims, starts, steps, zgrid.flatten())

# Interpolate
out = interpolator.eval(obs)

# Check result
assert np.allclose(out, zgrid.flatten(), rtol=1e-13)

# Serialize and deserialize
roundtrip_interpolator = interpn.MultilinearRegular.model_validate_json(
    interpolator.model_dump_json()
)
out2 = roundtrip_interpolator.eval(obs)

# Check result from roundtrip serialized/deserialized interpolator
assert np.all(out == out2)
```


# License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
