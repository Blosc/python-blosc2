# Release notes

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
  
