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

Just to whet your appetite, here are the results for an Apple M1 (2021)
with 8 GB of RAM but YMMV (and will vary!)::

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
      *** np.copyto() *** Time for memcpy():	0.030 s	(25.13 GB/s)

    Times for compressing/decompressing:

    *** the arange linear distribution ***
      *** blosclz , noshuffle  ***  0.275 s (2.71 GB/s) / 0.099 s (7.56 GB/s)	Compr. ratio:   2.0x
      *** blosclz , shuffle    ***  0.037 s (20.13 GB/s) / 0.024 s (30.92 GB/s)	Compr. ratio: 469.7x
      *** blosclz , bitshuffle ***  0.111 s (6.68 GB/s) / 0.237 s (3.15 GB/s)	Compr. ratio: 488.2x
      *** lz4     , noshuffle  ***  0.321 s (2.32 GB/s) / 0.069 s (10.88 GB/s)	Compr. ratio:   2.0x
      *** lz4     , shuffle    ***  0.034 s (21.89 GB/s) / 0.028 s (26.21 GB/s)	Compr. ratio: 279.2x
      *** lz4     , bitshuffle ***  0.121 s (6.18 GB/s) / 0.237 s (3.15 GB/s)	Compr. ratio:  87.7x
      *** lz4hc   , noshuffle  ***  2.250 s (0.33 GB/s) / 0.075 s (9.98 GB/s)	Compr. ratio:   2.0x
      *** lz4hc   , shuffle    ***  0.138 s (5.40 GB/s) / 0.047 s (15.87 GB/s)	Compr. ratio: 155.9x
      *** lz4hc   , bitshuffle ***  0.557 s (1.34 GB/s) / 0.167 s (4.46 GB/s)	Compr. ratio: 239.5x
      *** zlib    , noshuffle  ***  4.800 s (0.16 GB/s) / 0.275 s (2.71 GB/s)	Compr. ratio:   5.3x
      *** zlib    , shuffle    ***  0.219 s (3.41 GB/s) / 0.086 s (8.65 GB/s)	Compr. ratio: 273.8x
      *** zlib    , bitshuffle ***  0.336 s (2.22 GB/s) / 0.205 s (3.64 GB/s)	Compr. ratio: 457.9x
      *** zstd    , noshuffle  ***  2.887 s (0.26 GB/s) / 0.165 s (4.52 GB/s)	Compr. ratio:   7.9x
      *** zstd    , shuffle    ***  0.263 s (2.84 GB/s) / 0.032 s (23.08 GB/s)	Compr. ratio: 644.9x
      *** zstd    , bitshuffle ***  0.385 s (1.93 GB/s) / 0.156 s (4.77 GB/s)	Compr. ratio: 985.6x

    *** the linspace linear distribution ***
      *** blosclz , noshuffle  ***  0.376 s (1.98 GB/s) / 0.032 s (23.44 GB/s)	Compr. ratio:   1.0x
      *** blosclz , shuffle    ***  0.061 s (12.29 GB/s) / 0.035 s (21.47 GB/s)	Compr. ratio:  33.5x
      *** blosclz , bitshuffle ***  0.148 s (5.02 GB/s) / 0.241 s (3.09 GB/s)	Compr. ratio:  55.4x
      *** lz4     , noshuffle  ***  0.109 s (6.86 GB/s) / 0.038 s (19.86 GB/s)	Compr. ratio:   1.0x
      *** lz4     , shuffle    ***  0.051 s (14.56 GB/s) / 0.037 s (19.90 GB/s)	Compr. ratio:  40.5x
      *** lz4     , bitshuffle ***  0.136 s (5.49 GB/s) / 0.250 s (2.98 GB/s)	Compr. ratio:  59.5x
      *** lz4hc   , noshuffle  ***  3.611 s (0.21 GB/s) / 0.070 s (10.70 GB/s)	Compr. ratio:   1.1x
      *** lz4hc   , shuffle    ***  0.364 s (2.05 GB/s) / 0.036 s (20.61 GB/s)	Compr. ratio:  44.7x
      *** lz4hc   , bitshuffle ***  0.752 s (0.99 GB/s) / 0.158 s (4.70 GB/s)	Compr. ratio:  58.0x
      *** zlib    , noshuffle  ***  3.188 s (0.23 GB/s) / 0.489 s (1.52 GB/s)	Compr. ratio:   1.6x
      *** zlib    , shuffle    ***  0.393 s (1.90 GB/s) / 0.100 s (7.45 GB/s)	Compr. ratio:  44.6x
      *** zlib    , bitshuffle ***  0.519 s (1.44 GB/s) / 0.228 s (3.27 GB/s)	Compr. ratio:  66.9x
      *** zstd    , noshuffle  ***  3.567 s (0.21 GB/s) / 0.182 s (4.08 GB/s)	Compr. ratio:   1.2x
      *** zstd    , shuffle    ***  0.511 s (1.46 GB/s) / 0.056 s (13.36 GB/s)	Compr. ratio:  70.5x
      *** zstd    , bitshuffle ***  0.636 s (1.17 GB/s) / 0.202 s (3.68 GB/s)	Compr. ratio:  51.2x

    *** the random distribution ***
      *** blosclz , noshuffle  ***  0.373 s (2.00 GB/s) / 0.131 s (5.68 GB/s)	Compr. ratio:   2.1x
      *** blosclz , shuffle    ***  0.083 s (9.03 GB/s) / 0.029 s (25.30 GB/s)	Compr. ratio:   4.0x
      *** blosclz , bitshuffle ***  0.164 s (4.54 GB/s) / 0.238 s (3.13 GB/s)	Compr. ratio:   4.0x
      *** lz4     , noshuffle  ***  0.365 s (2.04 GB/s) / 0.060 s (12.46 GB/s)	Compr. ratio:   2.1x
      *** lz4     , shuffle    ***  0.076 s (9.74 GB/s) / 0.029 s (25.35 GB/s)	Compr. ratio:   4.0x
      *** lz4     , bitshuffle ***  0.154 s (4.83 GB/s) / 0.238 s (3.13 GB/s)	Compr. ratio:   4.6x
      *** lz4hc   , noshuffle  ***  2.039 s (0.37 GB/s) / 0.047 s (15.94 GB/s)	Compr. ratio:   2.8x
      *** lz4hc   , shuffle    ***  0.794 s (0.94 GB/s) / 0.051 s (14.65 GB/s)	Compr. ratio:   4.0x
      *** lz4hc   , bitshuffle ***  0.788 s (0.95 GB/s) / 0.172 s (4.33 GB/s)	Compr. ratio:   4.5x
      *** zlib    , noshuffle  ***  6.059 s (0.12 GB/s) / 0.423 s (1.76 GB/s)	Compr. ratio:   3.2x
      *** zlib    , shuffle    ***  0.977 s (0.76 GB/s) / 0.150 s (4.97 GB/s)	Compr. ratio:   4.7x
      *** zlib    , bitshuffle ***  0.955 s (0.78 GB/s) / 0.281 s (2.65 GB/s)	Compr. ratio:   4.6x
      *** zstd    , noshuffle  ***  4.085 s (0.18 GB/s) / 0.226 s (3.30 GB/s)	Compr. ratio:   4.0x
      *** zstd    , shuffle    ***  0.987 s (0.75 GB/s) / 0.061 s (12.15 GB/s)	Compr. ratio:   4.4x
      *** zstd    , bitshuffle ***  0.918 s (0.81 GB/s) / 0.150 s (4.96 GB/s)	Compr. ratio:   4.6x

As can be seen, in some situations it is perfectly possible to go faster than a plain memcpy().
Start using compression in your data workflows and feel the experience of doing more with less.


License
-------

The software is licenses under a 3-Clause BSD license. A copy of the
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
