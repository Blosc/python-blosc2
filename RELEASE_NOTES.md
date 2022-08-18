# Release notes

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

* Documentation of the new API is here: https://python-blosc2.readthedocs.io

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
  
* The `blosc.NOSHUFFLE` is replaced 
  by the `blosc2.NOFILTER`, but for backward 
  compatibility `blosc2.NOSHUFFLE` still exists.
  
* A bytearray or NumPy object can be passed to
the `blosc2.decompress` function to store the 
  decompressed data.
  
