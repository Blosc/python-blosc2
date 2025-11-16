# Release notes

## Changes from 3.11.0 to 3.11.1

* Change the `NDArray.size` to return the number of elements in array,
  instead of the size of the array in bytes. This follows the array
  API, so it is considered a fix, and takes precedence over a possible
  backward incompatibility.
* Tweak automatic chunk sizing of results for certain (e.g. linalg) operations
  to enhance performance
* Bug fixes for lazy expressions to allow a wider range of functionality
* Small bug fix for slice indexing with step larger than chunksize
* Various cosmetic fixes and streamlining (thanks to the indefatigable @DimitriPapadopoulos)

## Changes from 3.10.2 to 3.11.0

* Small optimisation for chunking in lazy expressions
* Extend Blosc2 computation machinery to accept general array inputs (PR #510)
* Refactoring and streamlining of get/setitem for non-unit steps (PR #513)
* Remote array testing now performed with `cat2cloud` (PR #511)
* Added argmax/argmin functions (PR #514)
* Change `squeeze` to return view (rather than modify array in-place) (PR #518)
* Modify `setitem` to load general array inputs into NDArrays (PR #517)

## Changes from 3.10.1 to 3.10.2

* LazyExpr.compute() now honors the `out` parameter for regular expressions (and not only for reductions).  See PR #506.

## Changes from 3.10.0 to 3.10.1

* Bumped to numexpr 2.14.1 to improve overflow behaviour for complex arguments for ``tanh`` and ``tanh``
* Bug fixes for lazy expression calculation
* Optimised computation for non-blosc2 chunked array arguments (e.g. Zarr, HDF5)
* Various cleanups and most importantly shipping of python 3.14 wheels due to @DimitriPapadopoulos!
* Now able to use blosc2 in AWS Lambda

## Changes from 3.9.1 to 3.10.0

* Improved documentation on thread management (thanks to [@orena1](@orena1) in PR #495)
* Enabled direct ingestion of Zarr arrays, and added examples for xarray ingestion
* Extended string-based lazy expression computation using a shape parser and modified lazy expression machinery so that expressions like "matmul(a, b) + c" can now be handled (PR #496).
* Streamlined inheritance from ``Operand`` to ensure access to basic methods like ``__add__`` for all computable objects (``NDArray``, ``LazyExpr``, ``LazyArray`` etc.) (PR ##500).

## Changes from 3.9.0 to 3.9.1

* Bumped to numexpr 2.13.1 to incorporate new maximum/minimum NaN handling and +/* for booleans
  which matches NumPy behaviour.
* Refactoring in order to ensure Blosc2 functions with NumPy 1.26.
* Streamlined documentation by introducing Array Protocol

## Changes from 3.8.0 to 3.9.0
Most changes come from PR #467 relating to array-api compliance.

* C-Blosc2 internal library updated to latest 2.21.3, increasing MAX_DIMS from 8 to 16

* numexpr version requirement pushed to 2.13.0 to incorporate
``round``, ``sign``, ``signbit``, ``copysign``, ``nextafter``, ``hypot``,
``maximum``, ``minimum``, ``trunc``, ``log2`` functions, as well as allow
integer outputs for certain functions when integr arguments are passed.
We also add floor division (``//``) and full dual bitwise (logical) AND, OR, XOR, NOT
support for integer (bool) arrays.

* Extended linear algebra functionality, offering generalised matrix multiplication
for arrays of arbitrary dimension via ``tensordot`` and an improved ``matmul``. In addition,
introduced ``vecdot``, ``diagonal`` and ``outer``, as well as useful indexing and associated functions such as ``take``, ``take_along_axis``, ``meshgrid`` and ``broadcast_to``.

* Added many ufuncs and methods (around 60) to ``NDArray`` to bring the library into further alignment with the array-api. Introduced a chunkwise lazyudf paradigm which is very powerful in order to implement ``clip`` and ``logaddexp``.

* Fixed a subtle but important bug for ``expand_dims`` (PR #479, PR #483) relating to reference counting for views.

## Changes from 3.7.2 to 3.8.0

* C-Blosc2 internal library updated to latest 2.21.2.

* numexpr version requirement pushed to 2.12.1 to incorporate
``isnan``, ``isfinite``, ``isinf`` functions.

* Indexing is now supported extensively and reasonably optimally for slices
with negative steps and general boolean arrays, with both get/setitem having
equal functionality. In PR #459 we extended the 1D fast path to general N-D,
with consequent speedups. In PR # we allowed fancy indexing and general slicing
with negative steps for set and getitem, with a memory-optimised path for setitem.

* Various attributes and methods for the ``NDArray`` class, as well as functions, have
been added to increase compliance with the array-api standard. In addition,
linspace and arange functions have been made more numerically stable and now strictly
comply even with difficult floating-point edge cases.

## Changes from 3.7.1 to 3.7.2

* C-Blosc2 internal library updated to latest 2.21.1.

* Revert signature of `TreeStore.__init__` for making benchmarks to get back
  to normal performance.

## Changes from 3.7.0 to 3.7.1

* Added `C2Array.slice()` method and `C2Array.nbytes`, `C2Array.cbytes`, `C2Array.cratio`, `C2Array.vlmeta` and `C2Array.info` properties (PR #455).

* Many usability improvements to the `TreeStore` class and friends.

* New section about `TreeStore` in basics NDArray tutorial.

* New blog post about `TreeStore` usage and performance at: https://www.blosc.org/posts/new-treestore-blosc2

* C-Blosc2 internal library updated to latest 2.21.0.

## Changes from 3.6.1 to 3.7.0

* Overhaul of documentation (API reference and Tutorials)

* Improvements to lazy expression indexing and in particular much more efficient memory usage when applying non-unit steps (PR #446).

* Extended functionality of ``expand_dims`` to match that of NumPy (note that this breaks the previous API) (PR #453).

* The biggest change is in the form of three new data storage classes (``EmbedStore``, ``DictStore`` and ``TreeStore``) which allow for the efficient storage of heterogeneous array data (PR #451). ``EmbedStore`` is essentially an ``SChunk`` wrapper which can be stored on-disk or in-memory; ``DictStore`` allows for mixed storage across memory, disk or indeed remote; and ``TreeStore`` is a hieracrhically-formatted version of ``DictStore`` which mimics the HDF5 file format. Write, access and storage performance are all very competitive with other packages - see [plots here](https://github.com/Blosc/python-blosc2/pull/451#issuecomment-3178828765).

## Changes from 3.6.0 to 3.6.1

* C-Blosc2 internal library updated to latest 2.19.1.

## Changes from 3.5.1 to 3.6.0

* Expose the `oindex` C-level functionality in Blosc2 for `NDArray`.

* Implement fancy indexing which closely matches NumPy functionality, using
`ndindex` library. Includes a fast path for 1D arrays, based on Zarr's implementation.

* A major refactoring of slicing for lazy expressions using `ndindex`. We have also
added support for slices with non-unit steps for reduction expressions, which has introduced
improvements that could be incorporated into other lazy expression machinery in the future.
More complex slicing is now supported.

* Minor bug fixes to ensure that Blosc2 indexing does not introduce dummy dimensions when NumPy does not,
and a more comprehensive `squeeze` function which squeezes specified dimensions.

## Changes from 3.5.0 to 3.5.1

* Reduced memory usage when computing slices of lazy expressions.
  This is a significant improvement for large arrays (up to 20x less).
  Also, we have added a fast path for slices that are small and fit in
  memory, which can be up to 20x faster than the previous implementation.
  See PR #430.

* `blosc2.concatenate()` has been renamed to `blosc2.concat()`.
  This is in line with the [Array API](https://data-apis.org/array-api).
  The old name is still available for backward compatibility, but it will
  be removed in a future release.

* Improve mode handling for concatenating to disk. See PR #428.
  Useful for concatenating arrays that are stored in disk, and allows
  specifying the mode to use when concatenating.

## Changes from 3.4.0 to 3.5.0

* New `blosc2.stack()` function for stacking multiple arrays along a new axis.
  Useful for creating multi-dimensional arrays from multiple 1D arrays.
  See PR #427. Thanks to [Luke Shaw](@lshaw8317) for the implementation!
  Blog: https://www.blosc.org/posts/blosc2-new-concatenate/#stacking-arrays

* New `blosc2.expand_dims()` function for expanding the dimensions of an array.
  This is useful for adding a new axis to an array, similar to NumPy's `np.expand_dims()`.
  See PR #427. Thanks to [Luke Shaw](@lshaw8317) for the implementation!

## Changes from 3.3.4 to 3.4.0

* Added C-level ``concatenate`` function in response to community request. When possible, uses an optimised path which avoids decompression and recompression, giving a significant performance boost. See PR #423.

* Slicing has been added to string-based lazyexprs, so that one may use
  expressions like `expr[1:3] +1` to compute a slice of the expression. This is useful
  for getting a sub-expression of a larger expression, and it works with both
  string-based and lazy expressions. See PR #417.

* Relatedly, the behaviour of the `slice` parameter in the `compute()` method of `LazyExpr` has been made more consistent and is now better documented, so that results are as expected. See PR #419.

* UDF support for pandas has been added to allow for the use of ``blosc2.jit``. See PR #418. Thanks to [@datapythonista](https://github.com/datapythonista) for the implementation!

## Changes from 3.3.3 to 3.3.4

* Expand possibilities for chaining string-based lazy expressions to incorporate
  data types which do not have shape attribute, e.g. int, float etc.
  See #406 and PR #411.

* Enable slicing within string-based lazy expressions. See PR #414.

* Improved casting for string-based lazy expressions.

* Documentation improvements, see PR #410.

* Compatibility fixes for working with `h5py` files.

## Changes from 3.3.2 to 3.3.3

* Expand possibilities for chaining string-based lazy expressions to include
  main operand types (LazyExpr and NDArray). Still have to incorporate other
  data types (which do not have shape attribute, e.g. int, float etc.).
  See #406.

* Fix indexing for lazy expressions, and allow use of None in getitem.
  See PR #402.

* Fix incorrect appending of dim to computed reductions. See PR #404.

* Fix `blosc2.linspace()` for incompatible num/shape.  See PR #408.

* Add support for NumPy dtypes that are n-dimensional (e.g.
  `np.dtype(("<i4,>f4", (10,))),`).

* New MAX_DIM constant for the maximum number of dimensions supported.
  This is useful for checking if a given array is too large to be handled.

* More refinements on guessing cache sizes for Linux.

* Update to C-Blosc2 2.17.2.dev.  Now, we are forcing the flush of modified
  pages only in write mode for mmap files. This fixes mmap issues on Windows.
  Thanks to @JanSellner for the implementation.

## Changes from 3.3.1 to 3.3.2

* Fixed a bug in the determination of chunk shape for the `NDArray` constructor.
  This was causing problems when creating `NDArray` instances with a CPU that
  was reporting a L3 cache size close (or exceeding) 2 GB.  See PR #392.

* Fixed a bug preventing the correct chaining of *string* lazy expressions for
  logical operators (`&`, `|`, `^`...).  See PR #391.

* More performance optimization for `blosc2.permute_dims`.  Thanks to
  Ricardo Sales Piquer (@ricardosp4) for the implementation.

* Now, storage defaults (`blosc2.storage_dflts`) are honored, even if no
  `storage=` param is used in constructors.

* We are distributing Python 3.10 wheels now.

## Changes from 3.3.0 to 3.3.1

* In our effort to better adapt to better adapt to the array API
  (https://data-apis.org/array-api/latest/), we have introduced
  permute_dims() and matrix_transpose() functions, and the .T property.
  This replaces to previous transpose() function, which is now deprecated.
  See PR #384.  Thanks to Ricardo Sales Piquer (@ricardosp4).

* Constructors like `arange()`, `linspace()` and `fromiter()` now
  use far less memory when creating large arrays. As an example, a 5 TB
  array of 8-byte floats now uses less than 200 MB of memory instead of
  170 GB previously.  See PR #387.

* Now, when opening a lazy expression with `blosc2.open()`, and there is
  a missing operand, the open still works, but the dtype and shape
  attributes are None.  This is useful for lazy expressions that have
  lost some operands, but you still want to open them for inspection.
  See PR #385.

* Added an example of getting a slice out of a C2Array.

## Changes from 3.2.1 to 3.3.0

* New `blosc2.transpose()` function for transposing 2D NDArray instances
  natively. See PR #375 and docs at
  https://www.blosc.org/python-blosc2/reference/autofiles/operations_with_arrays/blosc2.transpose.html#blosc2.transpose
  Thanks to Ricardo Sales Piquer (@ricardosp4) for the implementation.

* New fast path for `NDArray.slice()` for getting slices that are aligned with
  underlying chunks. This is a common operation when working with NDArray
  instances, and now it is up to 40x faster in our benchmarks (see PR #380).

* Returned `NDArray` object in `NDarray.slice()` now defaults to original
  codec/clevel/filters. The previous behavior was to use the default
  codec/clevel/filters.  See PR #378.  Thanks to Luke Shaw (@lshaw8317).

* Several English edits in the documentation.  Thanks to Luke Shaw (@lshaw8317)
  for his help in this area.

## Changes from 3.2.0 to 3.2.1

* The array containers are now using the `__array_interface__` protocol to
  expose the data in the array.  This allows for better interoperability with
  other libraries that support the `__array_interface__` protocol, like NumPy,
  CuPy, etc.  Now, the range of functions that can be used within the `blosc2.jit`
  decorator is way larger, and essentially all NumPy functions should work now.

  See examples at: https://github.com/Blosc/python-blosc2/blob/main/examples/ndarray/jit-numpy-funcs.py
  See benchmarks at: https://github.com/Blosc/python-blosc2/blob/main/bench/ndarray/jit-numpy-funcs.py

* The performance of constructors like `arange()`, `linspace()` and `fromiter()`
  has been improved.  Now, they can be up to 3x faster, specially with large
  arrays.

* C-Blosc2 updated to 2.17.1.  This fixes various UB as well as compiler warnings.

## Changes from 3.1.1 to 3.2.0

* Structured arrays can be larger than 255 bytes now.  This was a limitation
  in the previous versions, but now it is gone (the new limit is ~512 MB,
  which I hope will be enough for some time).

* New `blosc2.matmul()` function for computing matrix multiplication on NDArray
  instances.  This allows for efficient computations on compressed data that
  can be in-memory, on-disk and in the network.  See
  [here](https://www.blosc.org/python-blosc2/reference/autofiles/operations_with_arrays/blosc2.matmul.html)
  for more information.

* Support for building WASM32 wheels.  This is a new feature that allows to
  build wheels for WebAssembly 32-bit platforms.  This is useful for running
  Python code in the browser.

* Tested support for NumPy<2 (at least 1.26 series).  Now, the library should
  work with NumPy 1.26 and up.

* C-Blosc2 updated to 2.17.0.

* httpx has replaced by requests library for the remote proxy.  This has been
  done to avoid the need of the `httpx` library, which is not supported by
  Pyodide.

## Changes from 3.1.0 to 3.1.1

* Quick release to fix an issue with version number in the package (was reporting 3.0.0
  instead of 3.1.0).


## Changes from 3.0.0 to 3.1.0

### Improvements

* Optimizations for the compute engine. Now, it is faster and uses less memory.
  In particular, careful attention has been paid to the memory handling, as
  this is the main bottleneck for the compute engine in many instances.

* Improved detection of CPU cache sizes for Linux and macOS.  In particular,
  support for multi-CCX (AMD EPYC) and multi-socket systems has been implemented.
  Now, the library should be able to detect the cache sizes for most of the
  CPUs out there (specially on Linux).

* Optimization on NDArray slicing when the slice is a single chunk.  This is a
  common operation when working with NDArray instances, and now it is faster.

### New API functions and decorators

* New `blosc2.evaluate()` function for evaluating expressions on NDArray/NumPy
  instances.  This a drop-in replacement of `numexpr.evaluate()`, but with the
  next improvements:
  - More functionality than numexpr (e.g. reductions).
  - Follow casting rules of NumPy more closely.
  - Use both NumPy arrays and Blosc2 NDArrays in the same expression.

  See [here](https://www.blosc.org/python-blosc2/reference/autofiles/utilities/blosc2.evaluate.html)
  for more information.

* New `blosc2.jit` decorator for allowing NumPy expressions to be computed
  using the Blosc2 compute engine.  This is a powerful feature that allows for
  efficient computations on compressed data, and supports advanced features like
  reductions, filters and broadcasting.  See
  [here](https://www.blosc.org/python-blosc2/reference/autofiles/utilities/blosc2.jit.html)
  for more information.

* Support `out=` in `blosc2.mean()`, `blosc2.std()` and `blosc2.var()` reductions
  (besides `blosc2.sum()` and `blosc2.prod()`).


### Others

* Bumped to use latest C-Blosc2 sources (2.16.0).

* The cache for cpuinfo is now stored in `${HOME}/.cache/python-blosc2/cpuinfo.json`
  instead of `${HOME}/.blosc2-cpuinfo.json`; you can get rid of the latter, as
  the former is more standard (see PR #360). Thanks to Jonas Lundholm Bertelsen
  (@jonaslb).

## Changes from 3.0.0-rc.3 to 3.0.0

* A persistent cache for cpuinfo (stored in `$HOME/.blosc2-cpuinfo.json`) is
  now used to avoid repeated calls to the cpuinfo library.  This accelerates
  the startup time of the library considerably (up to 5x on my box).

* We should be creating conda packages now.  Thanks to @hmaarrfk for his
  assistance in this area.


## Changes from 3.0.0-rc.2 to 3.0.0-rc.3

* Now you can get and set the whole values of VLMeta instances with the `vlmeta[:]` syntax.
  The get part is syntactic sugar for `vlmeta.getall()` actually.

* `blosc2.copy()` now honors `cparams=` parameter.

* Now, compiling the package with `USE_SYSTEM_BLOSC2` envar set to `1` will use the
  system-wide Blosc2 library.  This is useful for creating packages that do not want
  to bundle the Blosc2 library (e.g. conda).

* Several changes in the build process to enable conda-forge packaging.

* Now, `blosc2.pack_tensor()` can pack empty tensors/arrays.  Fixes #290.


## Changes from 3.0.0-rc.1 to 3.0.0-rc.2

* Improved docs, tutorials and examples.  Have a look at our new docs at: https://www.blosc.org/python-blosc2.

* `blosc2.save()` is using `contiguous=True` by default now.

* `vlmeta[:]` is syntactic sugar for vlmeta.getall() now.

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
 Have a look at our new docs at: https://www.blosc.org/python-blosc2/
 For a guide on using UDFs, check out: https://www.blosc.org/python-blosc2/reference/autofiles/lazyarray/blosc2.lazyudf.html
 If interested in asynchronously fetching parts of an array, take a look at: https://www.blosc.org/python-blosc2/reference/autofiles/proxy/blosc2.Proxy.afetch.html
 Finally, there is a new tutorial on optimizing reductions in large NDArray objects: https://www.blosc.org/python-blosc2/getting_started/tutorials/04.reductions.html
 Special thanks @omaech and @martaiborrar for the excellent work on the documentation and examples, and to @NumFOCUS for their support in making this possible!

* New CParams, DParams and Storage dataclasses for better handling of parameters in the library.  Now, you can use these dataclasses to pass parameters to the library, and get a better error handling.  See [here](https://www.blosc.org/python-blosc2/reference/storage.html).  Thanks to @martaiborra for the excellent implementation.

* Better support for CParams in Proxy and C2Array instances.  This allows to better propagate compression parameters from Caterva2 datasets to the Proxy and C2Array instances, improving the perception of codecs and filters used originally in datasets.  Thanks to @FrancescAlted for the implementation.

* Many improvements in ruff linting and code style.  Thanks to @DimitriPapadopoulos for the excellent work in this area.


## Changes from 3.0.0-beta.1 to 3.0.0-beta.3

* Revamped documentation.  Now, the documentation is more complete and has a better structure. See [here](https://www.blosc.org/python-blosc2/).  Thanks to Oumaima Ech Chdig (@omaech), our newcomer to the Blosc team.  Also, thanks to NumFOCUS for the support in this task.

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
