=============
Python-Blosc2
=============

A Python wrapper for the extremely fast Blosc2 compression library
==================================================================

:Author: The Blosc development team
:Contact: blosc@blosc.org
:Github: https://github.com/Blosc/python-blosc2
:PyPi: |version|
:Gitter: |gitter|
:Code of Conduct: |Contributor Covenant|

.. |version| image:: https://img.shields.io/pypi/v/blosc2.png
        :target: https://pypi.python.org/pypi/blosc
.. |anaconda| image:: https://anaconda.org/conda-forge/python-blosc2/badges/version.svg
        :target: https://anaconda.org/conda-forge/python-blosc2
.. |gitter| image:: https://badges.gitter.im/Blosc/c-blosc.svg
        :target: https://gitter.im/Blosc/c-blosc
.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg
        :target: https://github.com/Blosc/community/blob/master/code_of_conduct.md


What it is
==========

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
==========

Blosc is now offering Python wheels for the main OS (Win, Mac and Linux) and platforms.
You can install binary packages from PyPi using ``pip``:

.. code-block:: console

    pip install blosc2

Documentation
=============

The documentation is here:

https://blosc.org/python-blosc2/python-blosc2.html

Also, some examples are available on:

https://github.com/Blosc/python-blosc2/tree/main/examples


Building
========

`python-blosc2` comes with the Blosc sources with it and can be built with:

.. code-block:: console

    git clone https://github.com/Blosc/python-blosc2/
    cd python-blosc2
    git submodule update --init --recursive
    python -m pip install -r requirements.txt
    python setup.py build_ext --inplace

That's all. You can proceed with testing section now.

Testing
=======

After compiling, you can quickly check that the package is sane by
running the doctests in ``blosc/test.py``:

.. code-block:: console

    python -m pip install -r requirements-tests.txt
    python -m pytest  (add -v for verbose mode)

Benchmarking
============

If curious, you may want to run a small benchmark that compares a plain
NumPy array copy against compression through different compressors in
your Blosc build:

.. code-block:: console

     PYTHONPATH=. python bench/compress_numpy.py

Just to whet your appetite, here are some speed figures for an Intel box (i9-10940X @ 3.30GHz)
with 64 GB RAM running Clear Linux::

    -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    python-blosc2 version: 0.1.7
    Blosc version: 2.0.0.rc2 ($Date:: 2021-05-26 #$)
    Compressors available: ['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']
    Compressor library versions:
      blosclz: 2.4.0
      lz4: 1.9.3
      lz4hc: 1.9.3
      zlib: 1.2.11.zlib-ng
      zstd: 1.5.0
    Python version: 3.7.9 (default, Aug 31 2020, 12:42:55)
    [GCC 7.3.0]
    Platform: Linux-5.12.6-1043.native-x86_64 (#1 SMP Sat May 22 04:04:10 PDT 2021)
    Linux dist: Clear Linux OS
    Processor: not recognized
    Byte-ordering: little
    Detected cores: 28
    Number of threads to use by default: 8
    -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    Creating NumPy arrays with 10**8 int64/float64 elements:
      *** np.copyto() *** Time for memcpy():	0.083 s	(8.93 GB/s)

    Times for compressing/decompressing:

    *** the arange linear distribution ***
      *** blosclz, noshuffle  ***  0.219 s (3.41 GB/s) / 0.083 s (8.93 GB/s)	cr:   2.0x
      *** blosclz, shuffle    ***  0.027 s (27.26 GB/s) / 0.035 s (21.38 GB/s)	cr: 469.7x
      *** blosclz, bitshuffle ***  0.078 s (9.56 GB/s) / 0.135 s (5.53 GB/s)	cr: 488.2x
      *** lz4    , noshuffle  ***  0.223 s (3.33 GB/s) / 0.075 s (9.92 GB/s)	cr:   2.0x
      *** lz4    , shuffle    ***  0.025 s (29.69 GB/s) / 0.035 s (21.18 GB/s)	cr: 279.2x
      *** lz4    , bitshuffle ***  0.079 s (9.43 GB/s) / 0.138 s (5.40 GB/s)	cr:  87.7x
      *** lz4hc  , noshuffle  ***  1.273 s (0.59 GB/s) / 0.076 s (9.85 GB/s)	cr:   2.0x
      *** lz4hc  , shuffle    ***  0.108 s (6.87 GB/s) / 0.032 s (23.37 GB/s)	cr: 155.9x
      *** lz4hc  , bitshuffle ***  0.359 s (2.08 GB/s) / 0.037 s (19.88 GB/s)	cr: 239.5x
      *** zlib   , noshuffle  ***  2.732 s (0.27 GB/s) / 0.146 s (5.09 GB/s)	cr:   5.3x
      *** zlib   , shuffle    ***  0.129 s (5.78 GB/s) / 0.046 s (16.11 GB/s)	cr: 273.8x
      *** zlib   , bitshuffle ***  0.179 s (4.17 GB/s) / 0.058 s (12.78 GB/s)	cr: 457.9x
      *** zstd   , noshuffle  ***  1.912 s (0.39 GB/s) / 0.113 s (6.61 GB/s)	cr:   7.9x
      *** zstd   , shuffle    ***  0.223 s (3.34 GB/s) / 0.031 s (24.18 GB/s)	cr: 644.9x
      *** zstd   , bitshuffle ***  0.242 s (3.07 GB/s) / 0.038 s (19.61 GB/s)	cr: 985.6x

    *** the linspace linear distribution ***
      *** blosclz, noshuffle  ***  0.099 s (7.55 GB/s) / 0.031 s (23.76 GB/s)	cr:   1.0x
      *** blosclz, shuffle    ***  0.050 s (15.02 GB/s) / 0.036 s (20.98 GB/s)	cr:  33.5x
      *** blosclz, bitshuffle ***  0.087 s (8.53 GB/s) / 0.147 s (5.08 GB/s)	cr:  55.4x
      *** lz4    , noshuffle  ***  0.085 s (8.77 GB/s) / 0.031 s (23.86 GB/s)	cr:   1.0x
      *** lz4    , shuffle    ***  0.038 s (19.53 GB/s) / 0.034 s (21.78 GB/s)	cr:  40.5x
      *** lz4    , bitshuffle ***  0.081 s (9.24 GB/s) / 0.146 s (5.09 GB/s)	cr:  59.5x
      *** lz4hc  , noshuffle  ***  1.902 s (0.39 GB/s) / 0.075 s (9.92 GB/s)	cr:   1.1x
      *** lz4hc  , shuffle    ***  0.237 s (3.14 GB/s) / 0.031 s (24.09 GB/s)	cr:  44.7x
      *** lz4hc  , bitshuffle ***  0.438 s (1.70 GB/s) / 0.035 s (21.03 GB/s)	cr:  58.0x
      *** zlib   , noshuffle  ***  2.078 s (0.36 GB/s) / 0.267 s (2.79 GB/s)	cr:   1.6x
      *** zlib   , shuffle    ***  0.239 s (3.11 GB/s) / 0.053 s (13.98 GB/s)	cr:  44.6x
      *** zlib   , bitshuffle ***  0.275 s (2.71 GB/s) / 0.065 s (11.45 GB/s)	cr:  66.9x
      *** zstd   , noshuffle  ***  2.792 s (0.27 GB/s) / 0.099 s (7.55 GB/s)	cr:   1.2x
      *** zstd   , shuffle    ***  0.374 s (1.99 GB/s) / 0.037 s (20.18 GB/s)	cr:  70.5x
      *** zstd   , bitshuffle ***  0.367 s (2.03 GB/s) / 0.053 s (14.10 GB/s)	cr:  51.2x

    *** the random distribution ***
      *** blosclz, noshuffle  ***  0.245 s (3.04 GB/s) / 0.105 s (7.12 GB/s)	cr:   2.1x
      *** blosclz, shuffle    ***  0.098 s (7.59 GB/s) / 0.038 s (19.56 GB/s)	cr:   4.0x
      *** blosclz, bitshuffle ***  0.163 s (4.57 GB/s) / 0.139 s (5.35 GB/s)	cr:   4.0x
      *** lz4    , noshuffle  ***  0.240 s (3.10 GB/s) / 0.040 s (18.65 GB/s)	cr:   2.1x
      *** lz4    , shuffle    ***  0.109 s (6.83 GB/s) / 0.039 s (19.28 GB/s)	cr:   4.0x
      *** lz4    , bitshuffle ***  0.144 s (5.18 GB/s) / 0.139 s (5.35 GB/s)	cr:   4.6x
      *** lz4hc  , noshuffle  ***  1.222 s (0.61 GB/s) / 0.035 s (21.25 GB/s)	cr:   2.8x
      *** lz4hc  , shuffle    ***  0.453 s (1.65 GB/s) / 0.038 s (19.66 GB/s)	cr:   4.0x
      *** lz4hc  , bitshuffle ***  0.419 s (1.78 GB/s) / 0.041 s (17.97 GB/s)	cr:   4.5x
      *** zlib   , noshuffle  ***  4.050 s (0.18 GB/s) / 0.208 s (3.58 GB/s)	cr:   3.2x
      *** zlib   , shuffle    ***  0.654 s (1.14 GB/s) / 0.074 s (10.06 GB/s)	cr:   4.7x
      *** zlib   , bitshuffle ***  0.610 s (1.22 GB/s) / 0.078 s (9.51 GB/s)	cr:   4.6x
      *** zstd   , noshuffle  ***  2.214 s (0.34 GB/s) / 0.125 s (5.95 GB/s)	cr:   4.0x
      *** zstd   , shuffle    ***  0.874 s (0.85 GB/s) / 0.039 s (19.01 GB/s)	cr:   4.4x
      *** zstd   , bitshuffle ***  0.858 s (0.87 GB/s) / 0.054 s (13.71 GB/s)	cr:   4.6x


For the matter of comparison, here are the results for an ARM box; an Apple MacBook Air M1 (2021)
with 8 GB of RAM::

    -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    python-blosc2 version: 0.1.6.dev0
    Blosc version: 2.0.0.rc2 ($Date:: 2021-05-26 #$)
    Compressors available: ['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']
    Compressor library versions:
      blosclz: 2.4.0
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
      *** np.copyto() *** Time for memcpy():	0.030 s	(25.04 GB/s)

    Times for compressing/decompressing:

    *** the arange linear distribution ***
      *** blosclz, noshuffle  ***  0.253 s (2.95 GB/s) / 0.109 s (6.83 GB/s)	cr:   2.0x
      *** blosclz, shuffle    ***  0.036 s (20.44 GB/s) / 0.024 s (31.08 GB/s)	cr: 469.7x
      *** blosclz, bitshuffle ***  0.123 s (6.04 GB/s) / 0.238 s (3.13 GB/s)	cr: 488.2x
      *** lz4    , noshuffle  ***  0.332 s (2.24 GB/s) / 0.072 s (10.39 GB/s)	cr:   2.0x
      *** lz4    , shuffle    ***  0.035 s (21.18 GB/s) / 0.030 s (24.93 GB/s)	cr: 279.2x
      *** lz4    , bitshuffle ***  0.126 s (5.91 GB/s) / 0.239 s (3.12 GB/s)	cr:  87.7x
      *** lz4hc  , noshuffle  ***  2.365 s (0.32 GB/s) / 0.080 s (9.35 GB/s)	cr:   2.0x
      *** lz4hc  , shuffle    ***  0.136 s (5.48 GB/s) / 0.047 s (15.89 GB/s)	cr: 155.9x
      *** lz4hc  , bitshuffle ***  0.545 s (1.37 GB/s) / 0.168 s (4.42 GB/s)	cr: 239.5x
      *** zlib   , noshuffle  ***  4.875 s (0.15 GB/s) / 0.279 s (2.67 GB/s)	cr:   5.3x
      *** zlib   , shuffle    ***  0.213 s (3.50 GB/s) / 0.091 s (8.20 GB/s)	cr: 273.8x
      *** zlib   , bitshuffle ***  0.344 s (2.16 GB/s) / 0.213 s (3.50 GB/s)	cr: 457.9x
      *** zstd   , noshuffle  ***  2.961 s (0.25 GB/s) / 0.168 s (4.44 GB/s)	cr:   7.9x
      *** zstd   , shuffle    ***  0.265 s (2.82 GB/s) / 0.035 s (21.46 GB/s)	cr: 644.9x
      *** zstd   , bitshuffle ***  0.392 s (1.90 GB/s) / 0.158 s (4.73 GB/s)	cr: 985.6x

    *** the linspace linear distribution ***
      *** blosclz, noshuffle  ***  0.372 s (2.00 GB/s) / 0.029 s (25.42 GB/s)	cr:   1.0x
      *** blosclz, shuffle    ***  0.065 s (11.46 GB/s) / 0.035 s (21.13 GB/s)	cr:  33.5x
      *** blosclz, bitshuffle ***  0.148 s (5.03 GB/s) / 0.250 s (2.98 GB/s)	cr:  55.4x
      *** lz4    , noshuffle  ***  0.109 s (6.84 GB/s) / 0.037 s (19.89 GB/s)	cr:   1.0x
      *** lz4    , shuffle    ***  0.052 s (14.27 GB/s) / 0.038 s (19.65 GB/s)	cr:  40.5x
      *** lz4    , bitshuffle ***  0.138 s (5.42 GB/s) / 0.250 s (2.99 GB/s)	cr:  59.5x
      *** lz4hc  , noshuffle  ***  3.962 s (0.19 GB/s) / 0.070 s (10.61 GB/s)	cr:   1.1x
      *** lz4hc  , shuffle    ***  0.366 s (2.04 GB/s) / 0.037 s (19.99 GB/s)	cr:  44.7x
      *** lz4hc  , bitshuffle ***  0.764 s (0.97 GB/s) / 0.159 s (4.69 GB/s)	cr:  58.0x
      *** zlib   , noshuffle  ***  3.290 s (0.23 GB/s) / 0.502 s (1.49 GB/s)	cr:   1.6x
      *** zlib   , shuffle    ***  0.403 s (1.85 GB/s) / 0.103 s (7.23 GB/s)	cr:  44.6x
      *** zlib   , bitshuffle ***  0.533 s (1.40 GB/s) / 0.228 s (3.27 GB/s)	cr:  66.9x
      *** zstd   , noshuffle  ***  3.747 s (0.20 GB/s) / 0.192 s (3.89 GB/s)	cr:   1.2x
      *** zstd   , shuffle    ***  0.483 s (1.54 GB/s) / 0.057 s (13.17 GB/s)	cr:  70.5x
      *** zstd   , bitshuffle ***  0.634 s (1.17 GB/s) / 0.204 s (3.65 GB/s)	cr:  51.2x

    *** the random distribution ***
      *** blosclz, noshuffle  ***  0.410 s (1.82 GB/s) / 0.135 s (5.50 GB/s)	cr:   2.1x
      *** blosclz, shuffle    ***  0.087 s (8.53 GB/s) / 0.029 s (25.29 GB/s)	cr:   4.0x
      *** blosclz, bitshuffle ***  0.169 s (4.40 GB/s) / 0.236 s (3.15 GB/s)	cr:   4.0x
      *** lz4    , noshuffle  ***  0.359 s (2.08 GB/s) / 0.060 s (12.50 GB/s)	cr:   2.1x
      *** lz4    , shuffle    ***  0.075 s (9.88 GB/s) / 0.029 s (25.40 GB/s)	cr:   4.0x
      *** lz4    , bitshuffle ***  0.155 s (4.81 GB/s) / 0.239 s (3.12 GB/s)	cr:   4.6x
      *** lz4hc  , noshuffle  ***  2.053 s (0.36 GB/s) / 0.045 s (16.71 GB/s)	cr:   2.8x
      *** lz4hc  , shuffle    ***  0.797 s (0.93 GB/s) / 0.051 s (14.63 GB/s)	cr:   4.0x
      *** lz4hc  , bitshuffle ***  0.795 s (0.94 GB/s) / 0.177 s (4.21 GB/s)	cr:   4.5x
      *** zlib   , noshuffle  ***  5.562 s (0.13 GB/s) / 0.367 s (2.03 GB/s)	cr:   3.2x
      *** zlib   , shuffle    ***  0.934 s (0.80 GB/s) / 0.148 s (5.03 GB/s)	cr:   4.7x
      *** zlib   , bitshuffle ***  0.959 s (0.78 GB/s) / 0.262 s (2.85 GB/s)	cr:   4.6x
      *** zstd   , noshuffle  ***  3.841 s (0.19 GB/s) / 0.228 s (3.27 GB/s)	cr:   4.0x
      *** zstd   , shuffle    ***  1.078 s (0.69 GB/s) / 0.069 s (10.76 GB/s)	cr:   4.4x
      *** zstd   , bitshuffle ***  1.044 s (0.71 GB/s) / 0.201 s (3.71 GB/s)	cr:   4.6x


As can be seen, is perfectly possible for python-blosc2 to go faster than a plain memcpy().

Start using compression in your data workflows and feel the experience of doing more with less!

License
=======

The software is licenses under a 3-Clause BSD license. A copy of the
python-blosc2 license can be found in `LICENSE <https://github.com/Blosc/python-blosc2/tree/main/LICENSE>`_. A copy of all licenses can be
found in `LICENSES/ <https://github.com/Blosc/python-blosc2/blob/main/LICENSES>`_.

Mailing list
============

Discussion about this module is welcome in the Blosc list:

blosc@googlegroups.com

http://groups.google.es/group/blosc

Twitter
=======

Please follow `@Blosc2 <https://twitter.com/Blosc2>`_ to get informed about the latest developments.

----

  **Enjoy data!**
