# interpn

Python bindings to the `interpn` Rust library for N-dimensional interpolation and extrapolation. 

## Features
| N-dimensional Grid Type →<br>↓ Interpolant Method | Regular | Rectilinear |
|-------------------|---------|-------------|
| Linear            |   ✅    |     ✅      |
| Cubic             |   ✅    |     💡      |

The methods provided here, while more limited in scope than scipy's, are
* significantly faster (1-2 orders of magnitude under most conditions)
* use almost no RAM (and perform no heap allocations at all)
* produce significantly improved floating-point error (by 1-2 orders of magnitude)
* are json-serializable using Pydantic
* can also be used easily in web and embedded applications via the Rust library

Docs: https://interpnpy.readthedocs.io/en/latest/

Repo: https://github.com/jlogan03/interpnpy

Rust library: https://github.com/jlogan03/interpn

# License
Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.