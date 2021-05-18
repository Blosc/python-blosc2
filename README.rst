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

    $ PYTHONPATH=. python bench/compress_numpy.py                                                                   (base)
    -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    python-blosc2 version: 0.1.6.dev0
    Blosc version: 2.0.0.rc.1 ($Date:: 2021-05-06 #$)
    Compressors available: ['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']
    Compressor library versions:
      blosclz: 2.3.0
      lz4: 1.9.3
      lz4hc: 1.9.3
      zlib: 1.2.11.zlib-ng
      zstd: 1.4.9
    Python version: 3.9.5 (default, May  3 2021, 19:12:05)
    [Clang 12.0.5 (clang-1205.0.22.9)]
    Platform: Darwin-20.4.0-arm64 (Darwin Kernel Version 20.4.0: Fri Mar  5 01:14:02 PST 2021; root:xnu-7195.101.1~3/RELEASE_ARM64_T8101)
    Processor: arm
    Byte-ordering: little
    Detected cores: 8
    Number of threads to use by default: 8
    -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    Creating NumPy arrays with 10**8 int64/float64 elements:
      *** ctypes.memmove() *** Time for memcpy():	0.134 s	(5.55 GB/s)

    Times for compressing/decompressing with clevel=5 and 8 threads

    *** the arange linear distribution ***
      *** blosclz , noshuffle  ***  0.374 s (1.99 GB/s) / 0.133 s (5.61 GB/s)	Compr. ratio:   1.0x
      *** blosclz , shuffle    ***  0.046 s (16.16 GB/s) / 0.037 s (20.02 GB/s)	Compr. ratio: 331.3x
      *** blosclz , bitshuffle ***  0.156 s (4.79 GB/s) / 0.162 s (4.61 GB/s)	Compr. ratio: 452.5x
      *** lz4     , noshuffle  ***  0.262 s (2.84 GB/s) / 0.081 s (9.15 GB/s)	Compr. ratio:   2.0x
      *** lz4     , shuffle    ***  0.024 s (31.58 GB/s) / 0.037 s (20.32 GB/s)	Compr. ratio: 268.0x
      *** lz4     , bitshuffle ***  0.144 s (5.16 GB/s) / 0.164 s (4.54 GB/s)	Compr. ratio:  87.2x
      *** lz4hc   , noshuffle  ***  1.050 s (0.71 GB/s) / 0.079 s (9.42 GB/s)	Compr. ratio:   2.0x
      *** lz4hc   , shuffle    ***  0.113 s (6.57 GB/s) / 0.049 s (15.24 GB/s)	Compr. ratio: 143.7x
      *** lz4hc   , bitshuffle ***  0.402 s (1.85 GB/s) / 0.174 s (4.28 GB/s)	Compr. ratio: 228.7x
      *** zlib    , noshuffle  ***  1.841 s (0.40 GB/s) / 0.246 s (3.03 GB/s)	Compr. ratio:   5.3x
      *** zlib    , shuffle    ***  0.327 s (2.28 GB/s) / 0.085 s (8.75 GB/s)	Compr. ratio: 232.3x
      *** zlib    , bitshuffle ***  0.446 s (1.67 GB/s) / 0.214 s (3.48 GB/s)	Compr. ratio: 375.4x
      *** zstd    , noshuffle  ***  2.196 s (0.34 GB/s) / 0.148 s (5.02 GB/s)	Compr. ratio:   7.9x
      *** zstd    , shuffle    ***  0.086 s (8.64 GB/s) / 0.043 s (17.49 GB/s)	Compr. ratio: 468.9x
      *** zstd    , bitshuffle ***  0.251 s (2.96 GB/s) / 0.172 s (4.34 GB/s)	Compr. ratio: 1005.5x

    *** the linspace linear distribution ***
      *** blosclz , noshuffle  ***  0.371 s (2.01 GB/s) / 0.111 s (6.70 GB/s)	Compr. ratio:   1.0x
      *** blosclz , shuffle    ***  0.103 s (7.25 GB/s) / 0.045 s (16.64 GB/s)	Compr. ratio:  23.7x
      *** blosclz , bitshuffle ***  0.240 s (3.10 GB/s) / 0.222 s (3.35 GB/s)	Compr. ratio:  26.9x
      *** lz4     , noshuffle  ***  0.200 s (3.72 GB/s) / 0.040 s (18.81 GB/s)	Compr. ratio:   1.0x
      *** lz4     , shuffle    ***  0.057 s (13.18 GB/s) / 0.049 s (15.20 GB/s)	Compr. ratio:  30.6x
      *** lz4     , bitshuffle ***  0.192 s (3.87 GB/s) / 0.200 s (3.73 GB/s)	Compr. ratio:  40.1x
      *** lz4hc   , noshuffle  ***  2.585 s (0.29 GB/s) / 0.093 s (8.04 GB/s)	Compr. ratio:   1.1x
      *** lz4hc   , shuffle    ***  0.276 s (2.70 GB/s) / 0.040 s (18.68 GB/s)	Compr. ratio:  23.8x
      *** lz4hc   , bitshuffle ***  0.663 s (1.12 GB/s) / 0.168 s (4.43 GB/s)	Compr. ratio:  31.8x
      *** zlib    , noshuffle  ***  2.455 s (0.30 GB/s) / 0.424 s (1.76 GB/s)	Compr. ratio:   1.6x
      *** zlib    , shuffle    ***  0.492 s (1.52 GB/s) / 0.102 s (7.32 GB/s)	Compr. ratio:  25.4x
      *** zlib    , bitshuffle ***  0.635 s (1.17 GB/s) / 0.231 s (3.23 GB/s)	Compr. ratio:  37.3x
      *** zstd    , noshuffle  ***  2.944 s (0.25 GB/s) / 0.167 s (4.46 GB/s)	Compr. ratio:   1.9x
      *** zstd    , shuffle    ***  0.233 s (3.20 GB/s) / 0.086 s (8.66 GB/s)	Compr. ratio:  34.2x
      *** zstd    , bitshuffle ***  0.354 s (2.10 GB/s) / 0.183 s (4.07 GB/s)	Compr. ratio:  48.7x

    *** the random distribution ***
      *** blosclz , noshuffle  ***  1.152 s (0.65 GB/s) / 0.067 s (11.19 GB/s)	Compr. ratio:   1.0x
      *** blosclz , shuffle    ***  0.120 s (6.20 GB/s) / 0.043 s (17.29 GB/s)	Compr. ratio:   4.0x
      *** blosclz , bitshuffle ***  0.175 s (4.26 GB/s) / 0.175 s (4.27 GB/s)	Compr. ratio:   4.0x
      *** lz4     , noshuffle  ***  0.242 s (3.08 GB/s) / 0.071 s (10.42 GB/s)	Compr. ratio:   2.5x
      *** lz4     , shuffle    ***  0.076 s (9.77 GB/s) / 0.047 s (16.00 GB/s)	Compr. ratio:   5.1x
      *** lz4     , bitshuffle ***  0.161 s (4.63 GB/s) / 0.164 s (4.53 GB/s)	Compr. ratio:   6.4x
      *** lz4hc   , noshuffle  ***  2.037 s (0.37 GB/s) / 0.050 s (14.92 GB/s)	Compr. ratio:   3.8x
      *** lz4hc   , shuffle    ***  0.690 s (1.08 GB/s) / 0.063 s (11.81 GB/s)	Compr. ratio:   5.2x
      *** lz4hc   , bitshuffle ***  0.379 s (1.97 GB/s) / 0.182 s (4.10 GB/s)	Compr. ratio:   6.2x
      *** zlib    , noshuffle  ***  1.390 s (0.54 GB/s) / 0.234 s (3.18 GB/s)	Compr. ratio:   4.2x
      *** zlib    , shuffle    ***  0.937 s (0.79 GB/s) / 0.105 s (7.10 GB/s)	Compr. ratio:   6.0x
      *** zlib    , bitshuffle ***  0.727 s (1.02 GB/s) / 0.218 s (3.42 GB/s)	Compr. ratio:   6.3x
      *** zstd    , noshuffle  ***  4.507 s (0.17 GB/s) / 0.208 s (3.59 GB/s)	Compr. ratio:   4.2x
      *** zstd    , shuffle    ***  1.153 s (0.65 GB/s) / 0.064 s (11.56 GB/s)	Compr. ratio:   6.0x
      *** zstd    , bitshuffle ***  0.292 s (2.55 GB/s) / 0.168 s (4.43 GB/s)	Compr. ratio:   6.4x

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
