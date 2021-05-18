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

    $ PYTHONPATH=.   (or "set PYTHONPATH=." on Win)
    $ export PYTHONPATH=.  (not needed on Win)
    $ pytest  (add -v for verbose mode)

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
      *** ctypes.memmove() *** Time for memcpy():	0.310 s	(2.40 GB/s)

    Times for compressing/decompressing with clevel=5 and 8 threads

    *** the arange linear distribution ***
      *** blosclz , noshuffle  ***  0.258 s (2.89 GB/s) / 0.178 s (4.18 GB/s)	Compr. ratio:   1.0x
      *** blosclz , shuffle    ***  0.055 s (13.51 GB/s) / 0.035 s (21.15 GB/s)	Compr. ratio: 331.3x
      *** blosclz , bitshuffle ***  0.155 s (4.82 GB/s) / 0.165 s (4.52 GB/s)	Compr. ratio: 452.5x
      *** lz4     , noshuffle  ***  0.340 s (2.19 GB/s) / 0.090 s (8.28 GB/s)	Compr. ratio:   2.0x
      *** lz4     , shuffle    ***  0.024 s (30.47 GB/s) / 0.037 s (19.90 GB/s)	Compr. ratio: 268.0x
      *** lz4     , bitshuffle ***  0.144 s (5.18 GB/s) / 0.172 s (4.34 GB/s)	Compr. ratio:  87.2x
      *** lz4hc   , noshuffle  ***  1.187 s (0.63 GB/s) / 0.090 s (8.25 GB/s)	Compr. ratio:   2.0x
      *** lz4hc   , shuffle    ***  0.114 s (6.56 GB/s) / 0.051 s (14.60 GB/s)	Compr. ratio: 143.7x
      *** lz4hc   , bitshuffle ***  0.399 s (1.87 GB/s) / 0.169 s (4.40 GB/s)	Compr. ratio: 228.7x
      *** zlib    , noshuffle  ***  1.833 s (0.41 GB/s) / 0.253 s (2.95 GB/s)	Compr. ratio:   5.3x
      *** zlib    , shuffle    ***  0.330 s (2.26 GB/s) / 0.087 s (8.52 GB/s)	Compr. ratio: 232.3x
      *** zlib    , bitshuffle ***  0.448 s (1.66 GB/s) / 0.214 s (3.48 GB/s)	Compr. ratio: 375.4x
      *** zstd    , noshuffle  ***  2.213 s (0.34 GB/s) / 0.149 s (5.00 GB/s)	Compr. ratio:   7.9x
      *** zstd    , shuffle    ***  0.086 s (8.65 GB/s) / 0.041 s (17.97 GB/s)	Compr. ratio: 468.9x
      *** zstd    , bitshuffle ***  0.245 s (3.05 GB/s) / 0.169 s (4.42 GB/s)	Compr. ratio: 1005.5x

    *** the linspace linear distribution ***
      *** blosclz , noshuffle  ***  0.460 s (1.62 GB/s) / 0.102 s (7.28 GB/s)	Compr. ratio:   1.0x
      *** blosclz , shuffle    ***  0.098 s (7.57 GB/s) / 0.044 s (17.03 GB/s)	Compr. ratio:  23.7x
      *** blosclz , bitshuffle ***  0.218 s (3.42 GB/s) / 0.179 s (4.15 GB/s)	Compr. ratio:  26.9x
      *** lz4     , noshuffle  ***  0.114 s (6.56 GB/s) / 0.040 s (18.77 GB/s)	Compr. ratio:   1.0x
      *** lz4     , shuffle    ***  0.045 s (16.72 GB/s) / 0.040 s (18.66 GB/s)	Compr. ratio:  30.6x
      *** lz4     , bitshuffle ***  0.155 s (4.81 GB/s) / 0.173 s (4.30 GB/s)	Compr. ratio:  40.1x
      *** lz4hc   , noshuffle  ***  2.526 s (0.29 GB/s) / 0.092 s (8.14 GB/s)	Compr. ratio:   1.1x
      *** lz4hc   , shuffle    ***  0.278 s (2.68 GB/s) / 0.038 s (19.46 GB/s)	Compr. ratio:  23.8x
      *** lz4hc   , bitshuffle ***  0.663 s (1.12 GB/s) / 0.156 s (4.77 GB/s)	Compr. ratio:  31.8x
      *** zlib    , noshuffle  ***  2.448 s (0.30 GB/s) / 0.426 s (1.75 GB/s)	Compr. ratio:   1.6x
      *** zlib    , shuffle    ***  0.487 s (1.53 GB/s) / 0.099 s (7.54 GB/s)	Compr. ratio:  25.4x
      *** zlib    , bitshuffle ***  0.632 s (1.18 GB/s) / 0.225 s (3.31 GB/s)	Compr. ratio:  37.3x
      *** zstd    , noshuffle  ***  2.696 s (0.28 GB/s) / 0.164 s (4.54 GB/s)	Compr. ratio:   1.9x
      *** zstd    , shuffle    ***  0.240 s (3.11 GB/s) / 0.086 s (8.68 GB/s)	Compr. ratio:  34.2x
      *** zstd    , bitshuffle ***  0.358 s (2.08 GB/s) / 0.195 s (3.82 GB/s)	Compr. ratio:  48.7x

    *** the random distribution ***
      *** blosclz , noshuffle  ***  1.050 s (0.71 GB/s) / 0.084 s (8.86 GB/s)	Compr. ratio:   1.0x
      *** blosclz , shuffle    ***  0.107 s (6.98 GB/s) / 0.043 s (17.42 GB/s)	Compr. ratio:   4.0x
      *** blosclz , bitshuffle ***  0.171 s (4.35 GB/s) / 0.164 s (4.54 GB/s)	Compr. ratio:   4.0x
      *** lz4     , noshuffle  ***  0.241 s (3.09 GB/s) / 0.066 s (11.29 GB/s)	Compr. ratio:   2.5x
      *** lz4     , shuffle    ***  0.073 s (10.23 GB/s) / 0.043 s (17.32 GB/s)	Compr. ratio:   5.1x
      *** lz4     , bitshuffle ***  0.159 s (4.70 GB/s) / 0.169 s (4.41 GB/s)	Compr. ratio:   6.4x
      *** lz4hc   , noshuffle  ***  2.098 s (0.36 GB/s) / 0.048 s (15.49 GB/s)	Compr. ratio:   3.8x
      *** lz4hc   , shuffle    ***  0.716 s (1.04 GB/s) / 0.060 s (12.39 GB/s)	Compr. ratio:   5.2x
      *** lz4hc   , bitshuffle ***  0.393 s (1.90 GB/s) / 0.189 s (3.95 GB/s)	Compr. ratio:   6.2x
      *** zlib    , noshuffle  ***  1.468 s (0.51 GB/s) / 0.236 s (3.15 GB/s)	Compr. ratio:   4.2x
      *** zlib    , shuffle    ***  0.964 s (0.77 GB/s) / 0.111 s (6.71 GB/s)	Compr. ratio:   6.0x
      *** zlib    , bitshuffle ***  0.776 s (0.96 GB/s) / 0.247 s (3.01 GB/s)	Compr. ratio:   6.3x
      *** zstd    , noshuffle  ***  5.001 s (0.15 GB/s) / 0.220 s (3.39 GB/s)	Compr. ratio:   4.2x
      *** zstd    , shuffle    ***  1.246 s (0.60 GB/s) / 0.064 s (11.58 GB/s)	Compr. ratio:   6.0x
      *** zstd    , bitshuffle ***  0.297 s (2.51 GB/s) / 0.183 s (4.08 GB/s)	Compr. ratio:   6.4x

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
