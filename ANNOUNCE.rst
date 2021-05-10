Announcing python-blosc2 0.1.1
==============================

Changes from python-blosc to python-blosc2
------------------------------------------

* The functions `compress_ptr` and `decompress_ptr` are replaced by pack and unpack since Pickle
  protocol 5 comes with out-of-band data.
* The function `pack_array` is equivalent to `pack`, which accepts any object with attributes `itemsize`
  and `size`.
* On the other hand, the function `unpack` doesn't return a numpy array whereas the `unpack_array`
  builds that array.
* The `blosc.NOSHUFFLE` is replaced by the `blosc2.NOFILTER`, but for backward
  compatibility `blosc2.NOSHUFFLE` still exists.
* A bytearray or NumPy object can be passed to the `blosc2.decompress` function to store the
  decompressed data.

In this release the package is compiled using the scikit-build tool
for a better integration with the C dependencies.

For more info, you can have a look at the release notes in:

https://github.com/Blosc/python-blosc2/releases

More docs and examples are available in the documentation site:

https://python-blosc2.rtfd.io


## What is it?

Blosc is an open source high performance compressor optimized for binary data
(i.e. floating point numbers, integers and booleans). It has
been designed to transmit data to the processor cache faster
than the traditional, non-compressed, direct memory fetch approach
via a memcpy() OS call. Blosc main goal is not just to reduce the
size of large datasets
on-disk or in-memory, but also to accelerate memory-bound computations.

python-blosc2 is a pythonic wrapper for the c-blosc2 library.


## Sources repository

The sources and documentation are managed through github services at:

http://github.com/Blosc/c-blosc2

c-blosc2 is distributed using the BSD license, see
[LICENSE](https://github.com/Blosc/c-blosc2/blob/master/LICENSE.txt)
for details.


## Mailing list

There is an official Blosc mailing list where discussions about
c-blosc2 are welcome:

blosc@googlegroups.com

http://groups.google.es/group/blosc


## Tweeter feed

Please follow @Blosc2 to get informed about the latest developments.


Enjoy Data!
- The Blosc Development Team