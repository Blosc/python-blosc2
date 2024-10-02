# Release notes

## Changes from 3.0.0-beta.3 to 3.0.0-beta.4

* Many new examples in the documentation.  Now, the documentation is more complete and has a better structure.
 Have a look at our new docs at: https://www.blosc.org/python-blosc2/index.html
 For a guide on using UDFs, check out: https://www.blosc.org/python-blosc2/reference/autofiles/lazyarray/blosc2.lazyudf.html
 If interested in asynchronously fetching parts of an array, take a look at: https://www.blosc.org/python-blosc2/reference/autofiles/proxy/blosc2.Proxy.afetch.html
 Finally, there is a new tutorial on optimizing reductions in large NDArray objects: https://www.blosc.org/python-blosc2/getting_started/tutorials/04.reductions.html
 Special thanks @omaech and @martaiborrar for the excellent work on the documentation and examples, and to @NumFOCUS for their support in making this possible!

* New CParams, DParams and Storage dataclasses for better handling of parameters in the library.  Now, you can use these dataclasses to pass parameters to the library, and get a better error handling.  See [here](https://www.blosc.org/python-blosc2/reference/storage.html).  Thanks to @martaiborra for the excellent implementation.

* Better support for CParams in Proxy and C2Array instances.  This allows to better propagate compression parameters from Caterva2 datasets to the Proxy and C2Array instances, improving the perception of codecs and filters used originally in datasets.  Thanks to @FrancescAlted for the implementation.

* Many improvements in ruff linting and code style.  Thanks to @DimitriPapadopoulos for the excellent work in this area.


## Changes from 3.0.0-beta.1 to 3.0.0-beta.3

* Revamped documentation.  Now, the documentation is more complete and has a better structure. See [here](https://www.blosc.org/python-blosc2/index.html).  Thanks to Oumaima Ech Chdig (@omaech), our newcomer to the Blosc team.  Also, thanks to NumFOCUS for the support in this task.

* New `Proxy` class to access other arrays, while providing caching. This is useful for example when you have a big array, and you want to access a small part of it, but you want to cache the accessed data for later use.  See [its doc](https://www.blosc.org/python-blosc2/reference/proxy.html).

* Lazy expressions can accept proxies as operands. 

* Read-ahead support for reading super-chunks from disk.  This allows for overlapping reads and computations, which can be a big performance boost for some workloads.

* New BLOSC_LOW_MEM envar for keeping memory under a minimum while evaluating expressions.  This makes it possible to evaluate expressions on very large arrays, even if the memory is limited (at the expense of performance).

* Fine tune block sizes for the internal compute engine.

* Better CPU cache size guessing for linux and macOS.

* Build tooling has been modernized and now uses `pyproject.toml` and `scikit-build-core` for managing dependencies and building the package.  Thanks to @LecrisUT for the excellent guidance in this area.

* Many code cleanup and syntax improvements in code.  Thanks to @DimitriPapadopoulos.


## Changes from 2.6.2 to 3.0.0-beta.1

* New evaluation engine (based on numexpr) for NDArray instances.  Now, you can evaluate expressions like `a + b + 1` where `a` and `b` are NDArray instances.  This is a powerful feature that allows for efficient computations on compressed data, and supports advanced features like reductions, filters, user-defined functions and broadcasting (still in beta).  See this [example](https://github.com/Blosc/python-blosc2/blob/main/examples/ndarray/eval_expr.py).

* As a consequence of the above, there are many new functions to operate with, and evaluate NDArray instances.  See the [function section docs](https://www.blosc.org/python-blosc2/reference/operations_with_arrays.html#functions) for more information.

* Support for NumPy 2.0.0 is here!  Now, the wheels are built with NumPy 2.0.0. If you want to use NumPy 1.x, you can still use it by installing NumPy 1.23 and up.

* Support for memory mapping in `SChunk` and `NDArray` instances.  This allows to map super-chunks stored in disk and access them as if they were in memory.  If curious, see  [some benchmarks here](https://github.com/Blosc/python-blosc2/blob/main/examples/ndarray/eval_expr.py).  Thanks to @JanSellner for the excellent implementation, both in the C and the Python libraries.

* Internal C-Blosc2 updated to 2.15.0.

* 32-bit platforms are officially unsupported now.  If you need support for 32-bit platforms, please use python-blosc 1.x series.

## Changes for 2.x series

* See the [release notes](https://github.com/Blosc/python-blosc2/blob/v2.x/RELEASE_NOTES.md) for the 2.x series.
