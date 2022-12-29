# Release notes

## Changes from 2.0.0 to 2.0.1

XXX version-specific blurb XXX


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

* No need to compile C-Blosc2 tests, benches or fuzzers.  Compilation time is much shorter now.

* Added `cratio`, `nbytes` and `cbytes` properties to `SChunk` instances.

* Added setters for `dparams` and `cparams` attributes in `SChunk`.


## Changes from 0.5.1 to 0.5.2

* Honor nested cparams properties in kwargs.

* C-Blosc2 upgraded to 2.4.3.  It should improve cratio for BloscLZ in combination with bitshuffle.

* Prefer pack_tensor/save_tensor in benches and examples


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
