# Changelog

## 0.2.4 2025-03-14

### Changed

* Update interpn rust dep
  * ~2-6x speedup for lower-dimensional inputs
* Update bindings for latest pyo3 and numpy rust deps
* Add .cargo/config.toml with configuration to enable vector instruction sets for x86 targets
  * ~10-30% speedup for regular grid methods on x86 machines (compounded with earlier 2-6x speedup)
* Regenerate CI configuration
* Support python 3.13

## 0.2.3 2024-08-20

### Changed

* Update interpn rust dep to 0.4.3 to capture upgraded linear methods

## 0.2.2 2024-07-08

### Changed

* Update python deps incl. numpy >2
* Update rust deps
* Support python 3.12
