# Release notes

## Changes from 2.7.0 to 2.7.1

* Updated to latest C-Blosc2 2.15.1.
  Fixes SIGKILL issues when using the `blosc2` library in old Intel CPUs.


## Changes from 2.6.2 to 2.7.0

* Updated to latest C-Blosc2 2.15.0.

* Deprecated `LazyExpr.evaluate()`.

* Fixed `_check_rc` function. See https://github.com/Blosc/python-blosc2/issues/187.


## Changes from 2.6.1 to 2.6.2

* Protection when platforms have just one CPU. This caused the
  internal number of threads to be 0, producing a division by zero.

* Updated to latest C-Blosc2 2.14.3.

## Changes from 2.6.0 to 2.6.1

* Updated to latest C-Blosc2 2.14.1. This was necessary to be able to
  load dynamics plugins on Windows.

## Changes from 2.5.1 to 2.6.0

* [EXP] New evaluation engine (based on numexpr) for NDArray instances.
  Now, you can evaluate expressions like `a + b + 1` where `a` and `b`
  are NDArray instances.  This is a powerful feature that allows for
  efficient computations on compressed data.  See this
  [example](https://github.com/Blosc/python-blosc2/blob/main/examples/ndarray/eval_expr.py)
  to see how this works.  Thanks to @omaech for her help in the `pow` function.

* As a consequence of the above, there are many new functions to operate with
  NDArray instances.  See the function section in
  [NDArray API](https://www.blosc.org/python-blosc2/reference/ndarray_api.html#functions)
  for more information.

* Support for NumPy 2.0.0 is here!  Now, the wheels are built with NumPy 2.0.0rc1.
  Please tell us in case you see any issues with this new version.

* Add `**kwargs` to `load_tensor()` function.  This allows to
  pass additional parameters to the deserialization function.
  Thanks to @jasam-sheja.

* Fix `vlmeta.to_dict()` not honoring tuple encoding.  Thanks to @ivilata.

* Check that chunks/blocks computation does not allow a 0 in blocks.
  Thanks to @ivilata.

* Many improvements in ruff rules and others.  Thanks to @DimitriPapadopoulos.

* Remove printing large arrays in notebooks (they use too much RAM in recent versions
  of Jupyter notebook).

* Updated to latest C-Blosc2 2.14.0.

## Changes from 2.5.0 to 2.5.1

* Updated to latest C-Blosc2 2.13.1.

* Fixed bug in `b2nd.h`.

## Changes from 2.4.0 to 2.5.0

* Updated to latest C-Blosc2 2.13.0.

* Added the filter `INT_TRUNC` for integer truncation.

* Added some optimizations for zstd.

* Now the grok library is initialized when loading the 
  plugin from C-Blosc2.

* Improved doc.

* Support for slices in ``blosc2.get_slice_nchunks()`` when using SChunk
  objects.

## Changes from 2.3.2 to 2.4.0

* Updated to latest C-Blosc2 2.12.0.

* Added `blosc2.get_slice_nchunks()` to get array of chunk 
  indexes needed to get a slice of a Blosc2 container.

* Added grok codec plugin.

* Added imported target with pkg-config to support windows.

## Changes from 2.3.1 to 2.3.2

* Support for `pathlib.Path` objects in all the places where `urlpath` is
  used (e.g. `blosc2.open()`). Thanks to Marta Iborra.

* Included docs for `SChunk.fill_special()` and `NDArray.dtype`. Thanks
  to Francesc Alted.

* Upgrade to latest C-Blosc2 2.11.3. It fixes a bug preventing the use of
  typesize > 255 in frames.  Now you can use a typesize up to 2**31-1.

## Changes from 2.3.0 to 2.3.1

* Temporarily disable AVX512 support in C-Blosc2 for wheels built by CI until
  run-time detection works properly.

## Changes from 2.2.9 to 2.3.0

* Require at least Cython 3 for building. Using previous versions worked but
  error handling was not correct (wheels were being built with Cython 3
  anyway).

* New `NDArray.to_cframe()` method and `blosc2.ndarray_from_cframe()` function
  for serializing and deserializing NDArrays to/from contiguous in-memory
  frames. Thanks to Francesc Alted.

* Add an optional `offset` argument to `blosc2.schunk.open()`, to access
  super-chunks stored in containers like HDF5. Thanks to Ivan Vilata.

* Assorted minor fixes to the blocksize/blockshape computation algorithm,
  avoiding some cases where it resulted in values exceeding maximum
  limits. Thanks to Ivan Vilata.

* Updated to latest C-Blosc2 2.11.2. It adds AVX512 support for the bitshuffle
  filter, fixes ARM and Raspberry Pi compatibility and assorted issues.

* Add python-blosc2 package definition for Guix. Thanks to Ivan Vilata.

## Changes from 2.2.8 to 2.2.9

* Support for specifying (plugable) tuner parameters in cparams. Thanks to
  Marta Iborra.

* Re-add support for Python 3.8.  Although we don't provide wheels for it,
  support for is there.

* Avoid duplicate iteration over the same dict.  Thanks to Dimitri Papadopoulos.

* Fix different issues with f-strings. Thanks to Dimitri Papadopoulos.

## Changes from 2.2.7 to 2.2.8

* Binary wheels for forthcoming Python 3.12 are available!

* Different improvements suggested by refurb and pyupgrade.
  Thanks to Dimitri Papadopoulos.

* Updated to latest C-Blosc2 2.10.4.

## Changes from 2.2.6 to 2.2.7

* Updated to latest C-Blosc2 2.10.3.

* Added openhtj2k codec plugin.

* Some small fixes regarding typos.


## Changes from 2.2.5 to 2.2.6

* Multithreading checks only apply to Python defined codecs and filters.
  Now it is possible to use multithreading with C codecs and filters plugins.
  See PR #127.

* New support for dynamic filters registry for Python.

* Now params for codec and filter plugins are correctly initialized
  when using `register_codec` and `register_filter` functions.

* Some fixes for Cython 3.0.0.  However,compatibility with Cython 3.0.0
  is not here yet, so build and install scripts are still requiring Cython<3.

* Updated to latest C-Blosc2 2.10.1.

## Changes from 2.2.4 to 2.2.5

* Updated to latest C-Blosc2 2.10.0.

* Use the new, fixed bytedelta filter introduced in C-Blosc2 2.10.0.

* Some small fixes in tutorials.


## Changes from 2.2.2 to 2.2.4

* Added a new [section of tutorials](https://www.blosc.org/python-blosc2/getting_started/tutorials.html)
  for a quick get start.

* Added a new [section on how to cite Blosc](https://github.com/Blosc/python-blosc2/tree/main#citing-blosc).

* New method `interchunks_info` for `SChunk` and `NDArray` classes.
  This iterates through chunks for getting meta info, like
  decompression ratio, whether the chunk is special or not, among
  others. For more information on how this works see
  [this example](https://github.com/Blosc/python-blosc2/blob/main/examples/ndarray/iterchunks_info.py).

* Now it is possible to register a dynamic plugin by passing None
  as the `encoder` and `decoder` arguments in the
  [register_codec](https://www.blosc.org/python-blosc2/reference/autofiles/top_level/blosc2.register_codec.html#blosc2.register_codec)
  function.

* Make shape of scalar slices NDArray objects to follow NumPy conventions.
  See [#117](https://github.com/Blosc/python-blosc2/issues/117).

* Updated to latest C-Blosc2 2.9.3.


## Changes from 2.2.2 to 2.2.3

* Updated to latest C-Blosc2 2.9.2.

* Better GIL handling.  Thanks to @martaiborra.


## Changes from 2.2.1 to 2.2.2

* Wheels are not including blosc2.pc (pkgconfig) anymore.  For details see:
  https://github.com/Blosc/python-blosc2/pull/111
  Thanks to @bnavigator for the PR.


## Changes from 2.2.0 to 2.2.1

* Updated to latest C-Blosc2 2.9.1.


## Changes from 2.1.1 to 2.2.0

* New bytedelta filter.  We have blogged about this: https://www.blosc.org/posts/bytedelta-enhance-compression-toolset/.  See the examples/ndarray/bytedelta_filter.py for a sample script.  We also have a short video on how bytedelta works: https://www.youtube.com/watch?v=5OXs7w2x6nw

* The compression defaults are changed to get a better balance between compression ratio, compression speed and decompression speed.  The new defaults are:

  - `cparams.typesize = 8`
  - `cparams.clevel = 1`
  - `cparams.compcode = Codec.ZSTD`
  - `filters = [Filter.SHUFFLE]`
  - `splitmode = SplitMode.ALWAYS_SPLIT`

  These changes have been based on the mentioned blog post above.

* `dtype.itemsize` will have preference over typesize in cparams (as it was documented).

* `blosc2.compressor_list(plugins=False)` do not list codec plugins by default now.  If you want to list plugins too, you need to pass `plugins=True`.

* Internal C-Blosc2 updated to latest version (2.8.0).


## Changes from 2.0.0 to 2.1.1

* New `NDArray` class for handling multidimensional arrays using compression. It includes:
  - Data type handling (fully compatible with NumPy)
  - Double partitioning

  See examples at: https://github.com/Blosc/python-blosc2/tree/main/examples/ndarray
  NDarray docs at: https://www.blosc.org/python-blosc2/reference/ndarray_api.html
  Explanatory video on why double partitioning: https://youtu.be/LvP9zxMGBng
  Also, see our blog on C-Blosc2 NDim counterpart: https://www.blosc.org/posts/blosc2-ndim-intro/

* Internal C-Blosc2 bumped to latest 2.7.1 version.


## Changes from 0.6.6 to 2.0.0

* Add support for user-defined filters and codecs.


## Changes from 0.6.5 to 0.6.6

* Add arm64 wheels for macosx (this time for real).


## Changes from 0.6.4 to 0.6.5

* Add arm64 wheels for macosx.


## Changes from 0.6.3 to 0.6.4

* Add arm64 wheels and remove musl builds (NumPy not having them makes the build process too long).

## Changes from 0.6.2 to 0.6.3

* Use oldest-supported-numpy for maximum compatibility.


## Changes from 0.6.1 to 0.6.2

* Updated C-Blosc2 to 2.6.0.


## Changes from 0.5.2 to 0.6.1

* Support for Python prefilters and postfilters.  With this, you can pre-process or post-process data in super-chunks automatically.  This machinery is handled internally by C-Blosc2, so it is very efficient (although it cannot work in multi-thread mode due to the GIL).  See the examples/ directory for different ways of using this.

* Support for fillers.  This is a specialization of a prefilter, and it allows to use Python functions to create new super-chunks from different kind of inputs (NumPy, SChunk instances, scalars), allowing computations among them and getting the result automatically compressed.  See a sample script in the examples/ directory.

* Lots of small improvements in the style, consistency and other glitches in the code.  Thanks to Dimitri Papadopoulos for hist attention to detail.

* No need to compile C-Blosc2 tests, benchmarks or fuzzers.  Compilation time is much shorter now.

* Added `cratio`, `nbytes` and `cbytes` properties to `SChunk` instances.

* Added setters for `dparams` and `cparams` attributes in `SChunk`.


## Changes from 0.5.1 to 0.5.2

* Honor nested cparams properties in kwargs.

* C-Blosc2 upgraded to 2.4.3.  It should improve cratio for BloscLZ in combination with bitshuffle.

* Prefer pack_tensor/save_tensor in benchmarks and examples


## Changes from 0.5.0 to 0.5.1

* Remove the testing of packing PyTorch or TensorFlow objects during wheels build.


## Changes from 0.4.1 to 0.5.0

* New `pack_tensor`, `unpack_tensor`, `save_tensor` and `load_tensor` functions for serializing/deserializing PyTorch and TensorFlow tensor objects.  They also understand NumPy arrays, so these are the new recommended ones for serialization.

* ``pack_array2`` do not modify the value of a possible `cparams` parameter anymore.

* The `pack_array2` / `save_array` have changed the serialization format to follow the new standard introduced in `pack_tensor`.  In the future `pack_array2` / `save_array` will probably be deprecated, so please change to `pack_tensor` / `save_tensor` as soon as you can.

* The new 'standard' for serialization relies on using the '__pack_tensor__' attribute as a `vlmeta` (variable length) metalayer.


## Changes from 0.4.0 to 0.4.1

* Add `msgpack` as a runtime requirement


## Changes from 0.3.2 to 0.4.0

* New `pack_array2()` and `unpack_array2()` functions for packing NumPy arrays.  Contrarily to `pack_array()` and `unpack_array()` counterparts, the new ones allow for compressing arrays larger than 2 GB in size.

* New `SChunk.to_cframe()` and `blosc2.from_cframe()` methods for serializing/deserializing `SChunk` instances.

* New `SChunk.get_slice()`, `SChunk.__getitem__()` and `SChunk.__setitem__()` methods for getting/setting slices from/to `SChunk` instances.

* The `compcode` parameter has been renamed to `codec`.  A `NameError` exception will be raised when using the old name.  Please update your code when you see this exception.

* More doc restructurings.  Hopefully, they are more pleasant to read now :-)


## Changes from 0.3.1 to 0.3.2

* Several leaks fixed.  Thanks to Christoph Gohlke.

* Internal C-Blosc2 updated to 2.3.1


## Changes from 0.3.0 to 0.3.1

* Internal C-Blosc2 updated to 2.3.0


## Changes from 0.2.0 to 0.3.0

* Added a new `blosc2.open(urlpath, mode)` function to be able to open persisted super-chunks.

* Added a new tutorial in notebook format (`examples/tutorial-basics.ipynb`) about the basics of python-blosc2.

* Internal C-Blosc2 updated to 2.2.0


## Changes from 0.1.10 to 0.2.0

* Internal C-Blosc updated to 2.0.4.


### Super-chunk implementation

* New `SChunk` class that allows to create super-chunks.
  This includes the capability of storing data in 4
  different ways (sparse/contiguous and in memory/on-disk),
  as well as storing variable length metalayers.

* Also, during the construction of a `SChunk` instance,
  an arbitrarily large data buffer can be given so that it is
  automatically split in chunks and those are appended to the
  `SChunk`.

* See `examples/schunk.py` and `examples/vlmeta.py` for some examples.

* Documentation of the new API is here: https://www.blosc.org/python-blosc2/python-blosc2.html

This release is the result of a grant offered by
the Python Software Foundation to Marta Iborra.
A blog entry was written describing the difficulties and relevant
aspects learned during the work:
https://www.blosc.org/posts/python-blosc2-initial-release/


## Changes from python-blosc2 0.1.9 to python-blosc2 0.1.10

* Release with C-Blosc 2.0.2 sources and binaries.


## Changes from python-blosc2 0.1.8 to python-blosc2 0.1.9

* Release with C-Blosc 2.0.1 sources and binaries.


## Changes from python-blosc2 0.1.7 to python-blosc2 0.1.8

* New versions of Blosc2 library added: plugins and lite.


## Changes from python-blosc2 0.1.5 to python-blosc2 0.1.7

* Headers and binaries for the C-Blosc2 library are starting
  to being distributed inside wheels.

* Internal C-Blosc2 submodule updated to 2.0.0-rc2.

* Repeating measurements 4 times in benchmarks so as to get more
  consistent figures.


## Changes from python-blosc2 0.1.1 to python-blosc2 0.1.5

* Fix some issues with packaging.  See:
  https://github.com/Blosc/python-blosc2/issues/9


## Changes from python-blosc to python-blosc2

* The functions `compress_ptr` and `decompress_ptr`
are replaced by pack and unpack since
Pickle protocol 5 comes with out-of-band data.

* The function `pack_array` is equivalent to `pack`,
which accepts any object with attributes `itemsize`
and `size`.

* On the other hand, the function `unpack` doesn't
return a numpy array whereas the `unpack_array`
builds that array.

* The `compcode` parameter has been renamed to `codec`.
A `NameError` exception will be raised when using the old name.
Please update your code when you see this exception.

* The different codecs are accessible via the `Codec` enumerated.
E.g. `Codec.LZ4` or `Codec.Zlib`

* The different filters are accessible via the `Filter` enumerated.
E.g. `Filter.SHUFFLE` or `Filter.BITSHUFFLE`

* The `blosc.NOSHUFFLE` is replaced by the `blosc2.Filter.NOFILTER`.

* A bytearray or NumPy object can be passed to the `blosc2.decompress`
function to store the decompressed data.
