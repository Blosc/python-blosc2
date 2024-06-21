# Release notes

## Changes from 3.0.0-beta.1 to 3.0.0-beta.2

XXX version-specific blurb XXX

## Changes from 2.6.2 to 3.0.0-beta.1

* New evaluation engine (based on numexpr) for NDArray instances.  Now, you can evaluate expressions like `a + b + 1` where `a` and `b` are NDArray instances.  This is a powerful feature that allows for efficient computations on compressed data, and supports advanced features like reductions, filters, user-defined functions and broadcasting (still in beta).  See this [example](https://github.com/Blosc/python-blosc2/blob/main/examples/ndarray/eval_expr.py).

* As a consequence of the above, there are many new functions to operate with, and evaluate NDArray instances.  See the [function section docs](https://www.blosc.org/python-blosc2/reference/operations_with_arrays.html#functions) for more information.

* Support for NumPy 2.0.0 is here!  Now, the wheels are built with NumPy 2.0.0. If you want to use NumPy 1.x, you can still use it by installing NumPy 1.23 and up.

* Support for memory mapping in `SChunk` and `NDArray` instances.  This allows to map super-chunks stored in disk and access them as if they were in memory.  If curious, see  [some benchmarks here](https://github.com/Blosc/python-blosc2/blob/main/examples/ndarray/eval_expr.py).  Thanks to @JanSellner for the excellent implementation, both in the C and the Python libraries.

* Internal C-Blosc2 updated to 2.15.0.

* 32-bit platforms are officially unsupported now.  If you need support for 32-bit platforms, please use python-blosc 1.x series.

## Changes for 2.x series

* See the [release notes](https://github.com/Blosc/python-blosc2/blob/v2.x/RELEASE_NOTES.md) for the 2.x series.
