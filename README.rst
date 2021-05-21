python-blosc2: a Python wrapper for the extremely fast Blosc2 compression library
=================================================================================

:Author: The Blosc development team
:Contact: blosc@blosc.org
:Github: https://github.com/Blosc/python-blosc2
:URL: http://python-blosc2.blosc.org
:PyPi: |version|
:Gitter: |gitter|
:Code of Conduct: |Contributor Covenant|

.. |version| image:: https://img.shields.io/pypi/v/blosc.png
        :target: https://pypi.python.org/pypi/blosc
.. |anaconda| image:: https://anaconda.org/conda-forge/python-blosc2/badges/version.svg
        :target: https://anaconda.org/conda-forge/python-blosc2
.. |gitter| image:: https://badges.gitter.im/Blosc/c-blosc.svg
        :target: https://gitter.im/Blosc/c-blosc
.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg
        :target: code_of_conduct.md


What it is
----------

Blosc (http://blosc.org) is a high performance compressor optimized for
binary data.  It has been designed to transmit data to the processor
cache faster than the traditional, non-compressed, direct memory fetch
approach via a memcpy() OS call.

Blosc works well for compressing numerical arrays that contains data
with relatively low entropy, like sparse data, time series, grids with
regular-spaced values, etc.

python-blosc2 is a Python package that wraps C-Blosc2, the newest version of
the Blosc compressor.  Currently python-blosc2 already reproduces the API of
python-blosc, so the former can be used as a drop-in replacement for the later.
However, there are a few exceptions for the complete compatibility that are listed
here:
https://github.com/Blosc/python-blosc2/blob/main/RELEASE_NOTES.md#changes-from-python-blosc-to-python-blosc2

In addition, python-blosc2 aims to leverage the new C-Blosc2 API so as to support
super-chunks, serialization and all the features introduced in C-Blosc2.
This is work in process and will be done incrementally in future releases.

**Note:** python-blosc2 is meant to be backward compatible with python-blosc data.
That means that it can read data generated with python-blosc, but the opposite
is not true (i.e. there is no *forward* compatibility).

Installing
----------

Blosc is now offering Python wheels for the main OS (Win, Mac and Linux) and platforms.
You can install binary packages from PyPi using ``pip``:

.. code-block:: console

    $ pip install blosc2

Documentation
-------------

The documentation is here:

https://python-blosc2.readthedocs.io/en/latest/

Also, some examples are available on python-blosc2 wiki page:

https://github.com/Blosc/python-blosc2/tree/main/examples

Lastly, here is the `recording
<https://www.youtube.com/watch?v=rilU44j_wUU&list=PLNkWzv63CorW83NY3U93gUar645jTXpJF&index=15>`_
and the `slides
<http://www.blosc.org/docs/haenel-ep14-compress-me-stupid.pdf>`_ from the talk
"Compress me stupid" at the EuroPython 2014.

Building
--------

`python-blosc2` comes with the Blosc sources with it and can be built with:

.. code-block:: console

    $ git clone https://github.com/Blosc/python-blosc2/
    $ cd python-blosc2
    $ git submodule update --init --recursive
    $ python -m pip install -r requirements.txt
    $ python setup.py build_ext --inplace

That's all. You can proceed with testing section now.

Testing
-------

After compiling, you can quickly check that the package is sane by
running the doctests in ``blosc/test.py``:

.. code-block:: console

    $ python -m pip install -r requirements-tests.txt
    $ python -m pytest  (add -v for verbose mode)

Benchmarking
------------

If curious, you may want to run a small benchmark that compares a plain
NumPy array copy against compression through different compressors in
your Blosc build:

.. code-block:: console

  $ PYTHONPATH=. python bench/compress_numpy.py

Just to whet your appetite, here are the results for an Apple M1
with 8 GB of RAM but YMMV (and will vary!)::

    $ PYTHONPATH=. python bench/compress_numpy.py
    -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    python-blosc2 version: 0.1.6.dev0
    Blosc version: 2.0.0.rc.2.dev ($Date:: 2021-05-06 #$)
    Compressors available: ['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']
    Compressor library versions:
      blosclz: 2.3.0
      lz4: 1.9.3
      lz4hc: 1.9.3
      zlib: 1.2.11.zlib-ng
      zstd: 1.5.0
    Python version: 3.9.5 (default, May  3 2021, 19:12:05)
    [Clang 12.0.5 (clang-1205.0.22.9)]
    Platform: Darwin-20.4.0-arm64 (Darwin Kernel Version 20.4.0: Fri Mar  5 01:14:02 PST 2021; root:xnu-7195.101.1~3/RELEASE_ARM64_T8101)
    Processor: arm
    Byte-ordering: little
    Detected cores: 8
    Number of threads to use by default: 8
    -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    Creating NumPy arrays with 10**8 int64/float64 elements:
      *** ctypes.memmove() *** Time for memcpy():	0.033 s	(22.57 GB/s)
    
    Times for compressing/decompressing with clevel=5 and 8 threads

    *** the arange linear distribution ***
      *** blosclz , noshuffle  ***  0.181 s (4.11 GB/s) / 0.169 s (4.40 GB/s)	Compr. ratio:   1.0x
      *** blosclz , shuffle    ***  0.028 s (26.94 GB/s) / 0.039 s (19.02 GB/s)	Compr. ratio: 331.3x
      *** blosclz , bitshuffle ***  0.090 s (8.23 GB/s) / 0.139 s (5.37 GB/s)	Compr. ratio: 450.4x
      *** lz4     , noshuffle  ***  0.290 s (2.57 GB/s) / 0.078 s (9.54 GB/s)	Compr. ratio:   2.0x
      *** lz4     , shuffle    ***  0.026 s (29.63 GB/s) / 0.039 s (19.23 GB/s)	Compr. ratio: 268.0x
      *** lz4     , bitshuffle ***  0.095 s (7.88 GB/s) / 0.137 s (5.45 GB/s)	Compr. ratio:  87.2x
      *** lz4hc   , noshuffle  ***  1.079 s (0.69 GB/s) / 0.077 s (9.70 GB/s)	Compr. ratio:   2.0x
      *** lz4hc   , shuffle    ***  0.099 s (7.52 GB/s) / 0.059 s (12.71 GB/s)	Compr. ratio: 143.7x
      *** lz4hc   , bitshuffle ***  0.367 s (2.03 GB/s) / 0.159 s (4.67 GB/s)	Compr. ratio: 228.7x
      *** zlib    , noshuffle  ***  1.553 s (0.48 GB/s) / 0.252 s (2.96 GB/s)	Compr. ratio:   5.3x
      *** zlib    , shuffle    ***  0.168 s (4.42 GB/s) / 0.086 s (8.64 GB/s)	Compr. ratio: 232.3x
      *** zlib    , bitshuffle ***  0.234 s (3.18 GB/s) / 0.176 s (4.24 GB/s)	Compr. ratio: 375.4x
      *** zstd    , noshuffle  ***  2.525 s (0.30 GB/s) / 0.140 s (5.31 GB/s)	Compr. ratio:   7.9x
      *** zstd    , shuffle    ***  0.205 s (3.63 GB/s) / 0.042 s (17.94 GB/s)	Compr. ratio: 468.9x
      *** zstd    , bitshuffle ***  0.317 s (2.35 GB/s) / 0.131 s (5.70 GB/s)	Compr. ratio: 1005.5x

    *** the linspace linear distribution ***
      *** blosclz , noshuffle  ***  0.615 s (1.21 GB/s) / 0.137 s (5.44 GB/s)	Compr. ratio:   1.0x
      *** blosclz , shuffle    ***  0.069 s (10.84 GB/s) / 0.045 s (16.73 GB/s)	Compr. ratio:  24.6x
      *** blosclz , bitshuffle ***  0.133 s (5.61 GB/s) / 0.141 s (5.27 GB/s)	Compr. ratio:  27.8x
      *** lz4     , noshuffle  ***  0.123 s (6.07 GB/s) / 0.038 s (19.36 GB/s)	Compr. ratio:   1.0x
      *** lz4     , shuffle    ***  0.046 s (16.32 GB/s) / 0.046 s (16.13 GB/s)	Compr. ratio:  30.6x
      *** lz4     , bitshuffle ***  0.105 s (7.09 GB/s) / 0.141 s (5.27 GB/s)	Compr. ratio:  40.1x
      *** lz4hc   , noshuffle  ***  2.299 s (0.32 GB/s) / 0.092 s (8.13 GB/s)	Compr. ratio:   1.1x
      *** lz4hc   , shuffle    ***  0.263 s (2.83 GB/s) / 0.047 s (15.96 GB/s)	Compr. ratio:  23.8x
      *** lz4hc   , bitshuffle ***  0.587 s (1.27 GB/s) / 0.127 s (5.88 GB/s)	Compr. ratio:  31.8x
      *** zlib    , noshuffle  ***  2.296 s (0.32 GB/s) / 0.438 s (1.70 GB/s)	Compr. ratio:   1.6x
      *** zlib    , shuffle    ***  0.336 s (2.22 GB/s) / 0.102 s (7.33 GB/s)	Compr. ratio:  25.4x
      *** zlib    , bitshuffle ***  0.427 s (1.74 GB/s) / 0.189 s (3.94 GB/s)	Compr. ratio:  37.3x
      *** zstd    , noshuffle  ***  2.473 s (0.30 GB/s) / 0.154 s (4.83 GB/s)	Compr. ratio:   1.9x
      *** zstd    , shuffle    ***  0.441 s (1.69 GB/s) / 0.084 s (8.85 GB/s)	Compr. ratio:  34.2x
      *** zstd    , bitshuffle ***  0.529 s (1.41 GB/s) / 0.165 s (4.53 GB/s)	Compr. ratio:  48.7x

    *** the random distribution ***
      *** blosclz , noshuffle  ***  1.021 s (0.73 GB/s) / 0.085 s (8.72 GB/s)	Compr. ratio:   1.1x
      *** blosclz , shuffle    ***  0.116 s (6.44 GB/s) / 0.044 s (16.78 GB/s)	Compr. ratio:   4.0x
      *** blosclz , bitshuffle ***  0.132 s (5.66 GB/s) / 0.135 s (5.54 GB/s)	Compr. ratio:   4.0x
      *** lz4     , noshuffle  ***  0.253 s (2.94 GB/s) / 0.068 s (10.98 GB/s)	Compr. ratio:   2.5x
      *** lz4     , shuffle    ***  0.080 s (9.35 GB/s) / 0.047 s (15.96 GB/s)	Compr. ratio:   5.1x
      *** lz4     , bitshuffle ***  0.111 s (6.73 GB/s) / 0.139 s (5.37 GB/s)	Compr. ratio:   6.4x
      *** lz4hc   , noshuffle  ***  2.341 s (0.32 GB/s) / 0.061 s (12.20 GB/s)	Compr. ratio:   3.8x
      *** lz4hc   , shuffle    ***  0.812 s (0.92 GB/s) / 0.066 s (11.27 GB/s)	Compr. ratio:   5.2x
      *** lz4hc   , bitshuffle ***  0.404 s (1.84 GB/s) / 0.181 s (4.12 GB/s)	Compr. ratio:   6.2x
      *** zlib    , noshuffle  ***  1.392 s (0.54 GB/s) / 0.249 s (2.99 GB/s)	Compr. ratio:   4.2x
      *** zlib    , shuffle    ***  0.820 s (0.91 GB/s) / 0.110 s (6.80 GB/s)	Compr. ratio:   6.0x
      *** zlib    , bitshuffle ***  0.548 s (1.36 GB/s) / 0.196 s (3.80 GB/s)	Compr. ratio:   6.3x
      *** zstd    , noshuffle  ***  4.362 s (0.17 GB/s) / 0.220 s (3.39 GB/s)	Compr. ratio:   4.2x
      *** zstd    , shuffle    ***  1.382 s (0.54 GB/s) / 0.063 s (11.88 GB/s)	Compr. ratio:   6.0x
      *** zstd    , bitshuffle ***  0.369 s (2.02 GB/s) / 0.145 s (5.13 GB/s)	Compr. ratio:   6.4x

In case you find your own results interesting, please report them back
to the authors!

License
-------

The software is licenses under a 3-Clause BSD licsense. A copy of the
python-blosc2 license can be found in `LICENSE <https://github.com/Blosc/python-blosc2/tree/main/LICENSE>`_. A copy of all licenses can be
found in `LICENSES/ <https://github.com/Blosc/python-blosc2/blob/main/LICENSES>`_.

Mailing list
------------

Discussion about this module is welcome in the Blosc list:

blosc@googlegroups.com

http://groups.google.es/group/blosc

Twitter fee
-----------

Please follow @Blosc2 to get informed about the latest developments.

----

  **Enjoy data!**
