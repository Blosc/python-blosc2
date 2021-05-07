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
==========

Blosc (http://blosc.org) is a high performance compressor optimized for
binary data.  It has been designed to transmit data to the processor
cache faster than the traditional, non-compressed, direct memory fetch
approach via a memcpy() OS call.

Blosc works well for compressing numerical arrays that contains data
with relatively low entropy, like sparse data, time series, grids with
regular-spaced values, etc.

python-blosc2 is a Python package that wraps C-Blosc2, the newest version of
the Blosc compressor.  python-blosc2 supports Python 3.7 or higher versions.

Installing
==========

Blosc is now offering Python wheels for the main OS (Win, Mac and Linux) and platforms. You can install binary packages from PyPi using ``pip``:

.. code-block:: console

    $ pip install blosc2

Documentation
=============

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
========

`python-blosc2` comes with the Blosc sources with it and can be built with:

.. code-block:: console

    $ git clone https://github.com/Blosc/python-blosc2/
    $ cd python-blosc2
    $ git submodule update --init --recursive
    $ python -m pip install -r requirements.txt
    $ python setup.py build_ext --inplace

That's all. You can proceed with testing section now.

Testing
=======

After compiling, you can quickly check that the package is sane by
running the doctests in ``blosc/test.py``:

.. code-block:: console

    $ PYTHONPATH=.   (or "set PYTHONPATH=." on Win)
    $ export PYTHONPATH=.  (not needed on Win)
    $ pytest  (add -v for verbose mode)

Benchmarking
============

If curious, you may want to run a small benchmark that compares a plain
NumPy array copy against compression through different compressors in
your Blosc build:

.. code-block:: console

  $ PYTHONPATH=. python bench/compress_numpy.py

Just to whet your appetite, here are the results for an Apple M1
with 8 GB of RAM but YMMV (and will vary!)::

    $ PYTHONPATH=. python bench/compress_numpy.py                                                                   (base)
    -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    python-blosc2 version: 0.1.0
    Blosc version: 2.0.0.rc.1 ($Date:: 2020-05-06 #$)
    Compressors available: ['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']
    Compressor library versions:
      blosclz: 2.3.0
      lz4: 1.9.3
      lz4hc: 1.9.3
      zlib: 1.2.11.zlib-ng
      zstd: 1.4.9
    Python version: 3.8.5 (default, Sep  4 2020, 02:22:02)
    [Clang 10.0.0 ]
    Platform: Darwin-20.4.0-x86_64 (Darwin Kernel Version 20.4.0: Fri Mar  5 01:14:02 PST 2021; root:xnu-7195.101.1~3/RELEASE_ARM64_T8101)
    Processor: i386
    Byte-ordering: little
    Detected cores: 8
    Number of threads to use by default: 8
    -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    Creating NumPy arrays with 10**8 int64/float64 elements:
      *** ctypes.memmove() *** Time for memcpy():	0.615 s	(1.21 GB/s)

    Times for compressing/decompressing with clevel=5 and 8 threads

    *** the arange linear distribution ***
      *** blosclz , nofilter   ***  0.423 s (1.76 GB/s) / 0.119 s (6.27 GB/s)	Compr. ratio:   1.0x
      *** blosclz , shuffle    ***  0.052 s (14.27 GB/s) / 0.043 s (17.48 GB/s)	Compr. ratio: 331.3x
      *** blosclz , bitshuffle ***  0.099 s (7.54 GB/s) / 0.116 s (6.41 GB/s)	Compr. ratio: 452.5x
      *** lz4     , nofilter   ***  0.437 s (1.70 GB/s) / 0.244 s (3.05 GB/s)	Compr. ratio:   2.0x
      *** lz4     , shuffle    ***  0.035 s (21.52 GB/s) / 0.047 s (15.89 GB/s)	Compr. ratio: 268.0x
      *** lz4     , bitshuffle ***  0.101 s (7.39 GB/s) / 0.125 s (5.97 GB/s)	Compr. ratio:  87.2x
      *** lz4hc   , nofilter   ***  1.502 s (0.50 GB/s) / 0.245 s (3.04 GB/s)	Compr. ratio:   2.0x
      *** lz4hc   , shuffle    ***  0.120 s (6.23 GB/s) / 0.054 s (13.77 GB/s)	Compr. ratio: 143.7x
      *** lz4hc   , bitshuffle ***  0.320 s (2.33 GB/s) / 0.136 s (5.49 GB/s)	Compr. ratio: 228.7x
      *** zlib    , nofilter   ***  1.972 s (0.38 GB/s) / 0.377 s (1.97 GB/s)	Compr. ratio:   5.3x
      *** zlib    , shuffle    ***  0.236 s (3.16 GB/s) / 0.126 s (5.90 GB/s)	Compr. ratio: 232.3x
      *** zlib    , bitshuffle ***  0.312 s (2.39 GB/s) / 0.198 s (3.76 GB/s)	Compr. ratio: 375.4x
      *** zstd    , nofilter   ***  2.802 s (0.27 GB/s) / 0.278 s (2.68 GB/s)	Compr. ratio:   7.9x
      *** zstd    , shuffle    ***  0.085 s (8.72 GB/s) / 0.090 s (8.32 GB/s)	Compr. ratio: 468.9x
      *** zstd    , bitshuffle ***  0.192 s (3.88 GB/s) / 0.145 s (5.13 GB/s)	Compr. ratio: 1005.5x

    *** the linspace linear distribution ***
      *** blosclz , nofilter   ***  0.781 s (0.95 GB/s) / 0.079 s (9.44 GB/s)	Compr. ratio:   1.0x
      *** blosclz , shuffle    ***  0.104 s (7.18 GB/s) / 0.046 s (16.15 GB/s)	Compr. ratio:  23.7x
      *** blosclz , bitshuffle ***  0.177 s (4.21 GB/s) / 0.145 s (5.15 GB/s)	Compr. ratio:  26.9x
      *** lz4     , nofilter   ***  0.271 s (2.75 GB/s) / 0.125 s (5.96 GB/s)	Compr. ratio:   1.0x
      *** lz4     , shuffle    ***  0.070 s (10.63 GB/s) / 0.054 s (13.75 GB/s)	Compr. ratio:  30.6x
      *** lz4     , bitshuffle ***  0.119 s (6.25 GB/s) / 0.139 s (5.36 GB/s)	Compr. ratio:  40.1x
      *** lz4hc   , nofilter   ***  2.980 s (0.25 GB/s) / 0.245 s (3.04 GB/s)	Compr. ratio:   1.1x
      *** lz4hc   , shuffle    ***  0.303 s (2.46 GB/s) / 0.053 s (13.96 GB/s)	Compr. ratio:  23.8x
      *** lz4hc   , bitshuffle ***  0.648 s (1.15 GB/s) / 0.127 s (5.86 GB/s)	Compr. ratio:  31.8x
      *** zlib    , nofilter   ***  3.045 s (0.24 GB/s) / 0.529 s (1.41 GB/s)	Compr. ratio:   1.6x
      *** zlib    , shuffle    ***  0.475 s (1.57 GB/s) / 0.127 s (5.85 GB/s)	Compr. ratio:  25.4x
      *** zlib    , bitshuffle ***  0.528 s (1.41 GB/s) / 0.222 s (3.36 GB/s)	Compr. ratio:  37.3x
      *** zstd    , nofilter   ***  3.633 s (0.21 GB/s) / 0.254 s (2.94 GB/s)	Compr. ratio:   1.9x
      *** zstd    , shuffle    ***  0.274 s (2.72 GB/s) / 0.111 s (6.71 GB/s)	Compr. ratio:  34.2x
      *** zstd    , bitshuffle ***  0.334 s (2.23 GB/s) / 0.168 s (4.43 GB/s)	Compr. ratio:  48.7x

    *** the random distribution ***
      *** blosclz , nofilter   ***  1.927 s (0.39 GB/s) / 0.064 s (11.72 GB/s)	Compr. ratio:   1.0x
      *** blosclz , shuffle    ***  0.281 s (2.65 GB/s) / 0.052 s (14.42 GB/s)	Compr. ratio:   4.0x
      *** blosclz , bitshuffle ***  0.178 s (4.20 GB/s) / 0.141 s (5.29 GB/s)	Compr. ratio:   4.0x
      *** lz4     , nofilter   ***  0.420 s (1.77 GB/s) / 0.142 s (5.24 GB/s)	Compr. ratio:   2.5x
      *** lz4     , shuffle    ***  0.120 s (6.22 GB/s) / 0.059 s (12.73 GB/s)	Compr. ratio:   5.1x
      *** lz4     , bitshuffle ***  0.144 s (5.19 GB/s) / 0.130 s (5.73 GB/s)	Compr. ratio:   6.4x
      *** lz4hc   , nofilter   ***  2.456 s (0.30 GB/s) / 0.148 s (5.04 GB/s)	Compr. ratio:   3.8x
      *** lz4hc   , shuffle    ***  0.820 s (0.91 GB/s) / 0.073 s (10.15 GB/s)	Compr. ratio:   5.2x
      *** lz4hc   , bitshuffle ***  0.381 s (1.96 GB/s) / 0.149 s (5.00 GB/s)	Compr. ratio:   6.2x
      *** zlib    , nofilter   ***  1.552 s (0.48 GB/s) / 0.329 s (2.26 GB/s)	Compr. ratio:   4.2x
      *** zlib    , shuffle    ***  0.970 s (0.77 GB/s) / 0.145 s (5.13 GB/s)	Compr. ratio:   6.0x
      *** zlib    , bitshuffle ***  0.678 s (1.10 GB/s) / 0.208 s (3.59 GB/s)	Compr. ratio:   6.3x
      *** zstd    , nofilter   ***  5.840 s (0.13 GB/s) / 0.322 s (2.31 GB/s)	Compr. ratio:   4.2x
      *** zstd    , shuffle    ***  1.399 s (0.53 GB/s) / 0.080 s (9.27 GB/s)	Compr. ratio:   6.0x
      *** zstd    , bitshuffle ***  0.253 s (2.95 GB/s) / 0.130 s (5.74 GB/s)	Compr. ratio:   6.4x

In case you find your own results interesting, please report them back
to the authors!

License
=======

The software is licenses under a 3-Clause BSD licsense. A copy of the
python-blosc2 license can be found in `LICENSE <LICENSE>`_. A copy of all licenses can be
found in `LICENSES/ <LICENSES/>`_.

Mailing list
============

Discussion about this module is welcome in the Blosc list:

blosc@googlegroups.com

http://groups.google.es/group/blosc

Twitter fee
===========

Please follow @Blosc2 to get informed about the latest developments.

----

  **Enjoy data!**
