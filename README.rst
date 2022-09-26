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

Just to whet your appetite, here are some speed figures for an AMD box (5950X, 16 cores)
running Ubuntu 22.04.  In particular, see how performance for `pack_array2/unpack_array2` has
improved vs the previous version (labeled as `pack_array/unpack_array`)::

    -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    python-blosc2 version: 0.3.3.dev0
    Blosc version: 2.4.2.dev ($Date:: 2022-09-16 #$)
    Compressors available: ['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']
    Compressor library versions:
      BLOSCLZ: 2.5.1
      LZ4: 1.9.4
      LZ4HC: 1.9.4
      ZLIB: 1.2.11.zlib-ng
      ZSTD: 1.5.2
    Python version: 3.9.13 | packaged by conda-forge | (main, May 27 2022, 16:56:21)
    [GCC 10.3.0]
    Platform: Linux-5.15.0-47-generic-x86_64 (#51-Ubuntu SMP Thu Aug 11 07:51:15 UTC 2022)
    Linux dist: Ubuntu 22.04.1 LTS
    Processor: x86_64
    Byte-ordering: little
    Detected cores: 16.0
    Number of threads to use by default: 8
    -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    Creating NumPy arrays with 10**8 int64/float64 elements:
      Time for copying array with np.copy:                  0.107 s (6.94 GB/s))


    *** the arange linear distribution ***
    Using *** Codec.BLOSCLZ *** compressor:
      Time for pack_array/unpack_array:     0.247/0.325 s (3.01/2.29 GB/s)) 	cr: 441.6x
      Time for pack_array2/unpack_array2:   0.033/0.137 s (22.34/5.42 GB/s)) 	cr: 444.0x
      Time for compress/decompress:         0.028/0.070 s (26.20/10.59 GB/s)) cr: 444.1x
    Using *** Codec.LZ4 *** compressor:
      Time for pack_array/unpack_array:     0.253/0.325 s (2.95/2.30 GB/s)) 	cr: 277.7x
      Time for pack_array2/unpack_array2:   0.033/0.133 s (22.87/5.62 GB/s)) 	cr: 279.2x
      Time for compress/decompress:         0.030/0.070 s (24.76/10.61 GB/s)) cr: 279.2x
    Using *** Codec.LZ4HC *** compressor:
      Time for pack_array/unpack_array:     0.309/0.320 s (2.41/2.33 GB/s)) 	cr: 155.4x
      Time for pack_array2/unpack_array2:   0.083/0.128 s (8.95/5.83 GB/s)) 	cr: 155.9x
      Time for compress/decompress:         0.077/0.062 s (9.66/12.01 GB/s)) 	cr: 155.9x
    Using *** Codec.ZLIB *** compressor:
      Time for pack_array/unpack_array:     0.311/0.322 s (2.40/2.31 GB/s)) 	cr: 273.3x
      Time for pack_array2/unpack_array2:   0.099/0.121 s (7.50/6.15 GB/s)) 	cr: 273.8x
      Time for compress/decompress:         0.095/0.066 s (7.87/11.32 GB/s)) 	cr: 273.8x
    Using *** Codec.ZSTD *** compressor:
      Time for pack_array/unpack_array:     0.374/0.315 s (1.99/2.36 GB/s)) 	cr: 630.8x
      Time for pack_array2/unpack_array2:   0.177/0.096 s (4.22/7.75 GB/s)) 	cr: 644.7x
      Time for compress/decompress:         0.167/0.062 s (4.46/12.07 GB/s)) 	cr: 644.9x

    *** the linspace linear distribution ***
    Using *** Codec.BLOSCLZ *** compressor:
      Time for pack_array/unpack_array:     0.263/0.320 s (2.83/2.33 GB/s)) 	cr:  35.7x
      Time for pack_array2/unpack_array2:   0.069/0.118 s (10.82/6.32 GB/s)) 	cr:  35.6x
      Time for compress/decompress:         0.041/0.068 s (18.35/11.01 GB/s))  cr:  35.6x
    Using *** Codec.LZ4 *** compressor:
      Time for pack_array/unpack_array:     0.263/0.320 s (2.84/2.33 GB/s)) 	cr:  40.5x
      Time for pack_array2/unpack_array2:   0.058/0.143 s (12.83/5.20 GB/s)) 	cr:  40.5x
      Time for compress/decompress:         0.037/0.068 s (19.87/10.97 GB/s))  cr:  40.5x
    Using *** Codec.LZ4HC *** compressor:
      Time for pack_array/unpack_array:     0.400/0.317 s (1.86/2.35 GB/s)) 	cr:  44.7x
      Time for pack_array2/unpack_array2:   0.208/0.109 s (3.57/6.86 GB/s)) 	cr:  44.7x
      Time for compress/decompress:         0.188/0.063 s (3.97/11.79 GB/s)) 	cr:  44.7x
    Using *** Codec.ZLIB *** compressor:
      Time for pack_array/unpack_array:     0.393/0.327 s (1.89/2.28 GB/s)) 	cr:  44.6x
      Time for pack_array2/unpack_array2:   0.216/0.130 s (3.44/5.71 GB/s)) 	cr:  44.6x
      Time for compress/decompress:         0.186/0.067 s (4.00/11.19 GB/s)) 	cr:  44.6x
    Using *** Codec.ZSTD *** compressor:
      Time for pack_array/unpack_array:     0.440/0.316 s (1.69/2.36 GB/s)) 	cr:  78.8x
      Time for pack_array2/unpack_array2:   0.259/0.107 s (2.88/6.98 GB/s)) 	cr:  78.8x
      Time for compress/decompress:         0.237/0.061 s (3.15/12.17 GB/s)) 	cr:  78.8x

    *** the random distribution ***
    Using *** Codec.BLOSCLZ *** compressor:
      Time for pack_array/unpack_array:     0.323/0.333 s (2.30/2.24 GB/s)) 	cr:   4.0x
      Time for pack_array2/unpack_array2:   0.425/0.106 s (1.75/7.04 GB/s)) 	cr:   4.0x
      Time for compress/decompress:         0.135/0.072 s (5.53/10.30 GB/s))   cr:   4.0x
    Using *** Codec.LZ4 *** compressor:
      Time for pack_array/unpack_array:     0.329/0.328 s (2.27/2.27 GB/s)) 	cr:   4.0x
      Time for pack_array2/unpack_array2:   0.430/0.154 s (1.73/4.83 GB/s)) 	cr:   4.0x
      Time for compress/decompress:         0.131/0.072 s (5.70/10.29 GB/s))   cr:   4.0x
    Using *** Codec.LZ4HC *** compressor:
      Time for pack_array/unpack_array:     0.663/0.331 s (1.12/2.25 GB/s)) 	cr:   4.0x
      Time for pack_array2/unpack_array2:   0.740/0.089 s (1.01/8.34 GB/s)) 	cr:   4.0x
      Time for compress/decompress:         0.435/0.068 s (1.71/10.98 GB/s)) 	cr:   4.0x
    Using *** Codec.ZLIB *** compressor:
      Time for pack_array/unpack_array:     0.683/0.340 s (1.09/2.19 GB/s)) 	cr:   4.7x
      Time for pack_array2/unpack_array2:   0.748/0.122 s (1.00/6.09 GB/s)) 	cr:   4.7x
      Time for compress/decompress:         0.469/0.083 s (1.59/9.00 GB/s)) 	cr:   4.7x
    Using *** Codec.ZSTD *** compressor:
      Time for pack_array/unpack_array:     0.667/0.322 s (1.12/2.31 GB/s)) 	cr:   4.4x
      Time for pack_array2/unpack_array2:   0.860/0.099 s (0.87/7.54 GB/s)) 	cr:   4.4x
      Time for compress/decompress:         0.478/0.066 s (1.56/11.23 GB/s)) 	cr:   4.4x

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
