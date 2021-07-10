# Release notes

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

* Repeating measurements 4 times in benchs so as to get more
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
  
