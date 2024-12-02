# Release notes

## Changes from 3.0.0-rc.1 to 3.0.0

* Improved docs, tutorials and examples.  Have a look at our new docs at: https://www.blosc.org/python-blosc2.

* `blosc2.save()` is using `contiguous=True` by default now.

* `vlmeta[:]` is syntatic sugar for vlmeta.getall() now.

* Add `NDArray.meta` property as a proxy to `NDArray.shunk.vlmeta`.

* Reductions over single fields in structured NDArrays are now supported.  For example, given an array `sarr` with fields 'a', 'b' and 'c', `sarr["a"]["b >= c"].std()` returns the standard deviation of the values in field 'a' for the rows that fulfills that values in fields in 'b' are larger than values in 'c' (`b >= c` above).

* As per discussion #337, the default of cparams.splitmode is now AUTO_SPLIT. See #338 though.


## Changes from 3.0.0-beta.4 to 3.0.0-rc.1

### General improvements

* New ufunc support for NDArray instances. Now, you can use NumPy ufuncs on NDArray instances, and mix them with other NumPy arrays. This is a powerful feature that allows for more interoperability with NumPy.

* Enhanced dtype inference, so that it mimics now more NumPy than the numexpr one. Although perfect adherence to NumPy casting conventions is not there yet, it is a big step forward towards better compatibility with NumPy.

* Fix dtype for sum and prod reductions. Now, the dtype of the result of a sum or prod reduction is the same as the input array, unless the dtype is not supported by the reduction, in which case the dtype is promoted to a supported one. It is more NumPy-like now.

* Many improvements on the computation of UDFs (User Defined Functions). Now, the lazy UDF computation is way more robust and efficient.

* Support reductions inside queries in structured NDArrays. For example, given an array `sarr` with fields 'a', 'b' and 'c', the next `farr = sarr["b >= c"].sum("a").compute()` puts in `farr` the sum of the values in field 'a' for the rows that fulfills that values in fields in 'b' are larger than values in 'c' (b >= c above).

* Implemented combining data filtering, as well as sorting, in structured NDArrays. For example, given an array `sarr` with fields 'a', 'b' and 'c', the next `farr = sarr["b >= c"].indices(order="c").compute()` puts in farr the indices of the rows that fulfills that values in fields in 'b' are larger than values in 'c' (`b >= c` above), ordered by column 'c'.

* Reductions can be stored in persistent lazy expressions. Now, if you have a lazy expression that contains a reduction, the result of the reduction is preserved in the expression, so that you can reuse it later on. See https://www.blosc.org/posts/persistent-reductions/ for more information.

* Many improvements in ruff linting and code style. Thanks to @DimitriPapadopoulos for the excellent work in this area.

### API changes

* `LazyArray.eval()` has been renamed to `LazyArray.compute()`. This avoids confusion with the `eval()` function in Python, and it is more in line with the Dask API.

This is the main change in the API that is not backward compatible with previous beta. If you have code that still uses `LazyArray.eval()`, you should change it to `LazyArray.compute()`.  Starting from this release, the API will be stable and backward compatibility will be maintained.

### New API calls

* New `reshape()` function and `NDArray.reshape()` method allow to do efficient reshaping between NDArrays that follows C order. Only 1-dim -> n-dim is currently supported though.

* `New NDArray.__iter__()` iterator following NumPy conventions.

* Now, `NDArray.__getitem__()` supports (n-dim) bool arrays or sequences of integers as indices (only 1-dim for now). This follows NumPy conventions.

* A new `NDField.__setitem__()` has been added to allow for setting values in a structured NDArray.

* `struct_ndarr['field']` now works as in NumPy, that is, it returns an array with the values in 'field' in the structured NDArray.

* Several new constructors are available for creating NDArray instances, like `arange()`, `linspace()` and `fromiter()`. These constructors leverage the internal `lazyudf()` function and make it easier to create NDArray instances from scratch. See e.g. https://github.com/Blosc/python-blosc2/blob/main/examples/ndarray/arange-constructor.py for an example.

* Structured LazyArrays received a new `.indices()` method that returns the indices of the elements that fulfill a condition. When combined with the new support of list of indices as key for `NDArray.__getitem__()`, this is useful for creating indexes for data.  See https://github.com/Blosc/python-blosc2/blob/main/examples/ndarray/filter_sort_fields.py for an example.

* LazyArrays received a new `.sort()` method that sorts the elements in the array.  For example, given an array `sarr` with fields 'a', 'b' and 'c', the next `farr = sarr["b >= c"].sort("c").compute()` puts in `farr` the rows that fulfills that values in fields in 'b' are larger than values in 'c' (`b >= c` above), ordered by column 'c'.

* New `expr_operands()` function for extracting operands from a string expression.

* New `validate_expr()` function for validating a string expression.

* New `CParams`, `DParams` and `Storage` dataclasses for better handling of parameters in the library. Now, you can use these dataclasses to pass parameters to the library, and get a better error handling. Thanks to @martaiborra for the excellent implementation and @omaech for revamping docs and examples to use them.  See e.g. https://www.blosc.org/python-blosc2/getting_started/tutorials/02.lazyarray-expressions.html.

### Documentation improvements

* Much improved documentation on how to efficiently compute with compressed NDArray data. Documentation updates highlight these features and improve usability for new users. Thanks to @omaech and @martaiborra for their excellent work on the documentation and examples, and to @NumFOCUS for their support in making this possible!  See https://www.blosc.org/python-blosc2/getting_started/tutorials/04.reductions.html for an example.

* New remote proxy tutorial. This tutorial shows how to use the Proxy class to access remote arrays, while providing caching. https://www.blosc.org/python-blosc2/getting_started/tutorials/06.remote_proxy.html . Thanks to @omaech for her work on this tutorial.

* New tutorial on "Mastering Persistent, Dynamic Reductions and Lazy Expressions". See https://www.blosc.org/posts/persistent-reductions/


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
