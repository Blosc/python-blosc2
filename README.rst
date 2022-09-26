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
      Time for copying array with np.copy:                  0.104 s (7.15 GB/s))


    *** the arange linear distribution ***
    Using *** Codec.BLOSCLZ *** compressor:
      Time for pack_array/unpack_array:     0.242/0.337 s (3.08/2.21 GB/s)) 	cr: 441.6x
      Time for pack_array2/unpack_array2:   0.033/0.120 s (22.46/6.21 GB/s)) 	cr: 444.0x
      Time for compress/decompress:         0.028/0.076 s (26.63/9.84 GB/s)) 	cr: 444.1x
    Using *** Codec.LZ4 *** compressor:
      Time for pack_array/unpack_array:     0.243/0.324 s (3.07/2.30 GB/s)) 	cr: 277.7x
      Time for pack_array2/unpack_array2:   0.032/0.134 s (23.51/5.57 GB/s)) 	cr: 279.2x
      Time for compress/decompress:         0.030/0.067 s (24.90/11.08 GB/s)) 	cr: 279.2x
    Using *** Codec.LZ4HC *** compressor:
      Time for pack_array/unpack_array:     0.311/0.319 s (2.39/2.33 GB/s)) 	cr: 155.4x
      Time for pack_array2/unpack_array2:   0.084/0.121 s (8.92/6.15 GB/s)) 	cr: 155.9x
      Time for compress/decompress:         0.076/0.062 s (9.74/11.97 GB/s)) 	cr: 155.9x
    Using *** Codec.ZLIB *** compressor:
      Time for pack_array/unpack_array:     0.338/0.325 s (2.20/2.30 GB/s)) 	cr: 273.3x
      Time for pack_array2/unpack_array2:   0.100/0.126 s (7.46/5.90 GB/s)) 	cr: 273.8x
      Time for compress/decompress:         0.095/0.066 s (7.84/11.35 GB/s)) 	cr: 273.8x
    Using *** Codec.ZSTD *** compressor:
      Time for pack_array/unpack_array:     0.373/0.316 s (2.00/2.36 GB/s)) 	cr: 630.8x
      Time for pack_array2/unpack_array2:   0.185/0.094 s (4.02/7.93 GB/s)) 	cr: 644.7x
      Time for compress/decompress:         0.167/0.062 s (4.46/12.04 GB/s)) 	cr: 644.9x

    *** the linspace linear distribution ***
    Using *** Codec.BLOSCLZ *** compressor:
      Time for pack_array/unpack_array:     0.269/0.322 s (2.77/2.32 GB/s)) 	cr:  35.7x
      Time for pack_array2/unpack_array2:   0.071/0.144 s (10.56/5.17 GB/s)) 	cr:  35.6x
      Time for compress/decompress:         0.042/0.067 s (17.74/11.19 GB/s)) 	cr:  35.6x
    Using *** Codec.LZ4 *** compressor:
      Time for pack_array/unpack_array:     0.253/0.317 s (2.94/2.35 GB/s)) 	cr:  40.5x
      Time for pack_array2/unpack_array2:   0.060/0.151 s (12.46/4.94 GB/s)) 	cr:  40.5x
      Time for compress/decompress:         0.037/0.065 s (20.16/11.52 GB/s)) 	cr:  40.5x
    Using *** Codec.LZ4HC *** compressor:
      Time for pack_array/unpack_array:     0.402/0.319 s (1.85/2.34 GB/s)) 	cr:  44.7x
      Time for pack_array2/unpack_array2:   0.210/0.104 s (3.55/7.13 GB/s)) 	cr:  44.7x
      Time for compress/decompress:         0.188/0.063 s (3.97/11.90 GB/s)) 	cr:  44.7x
    Using *** Codec.ZLIB *** compressor:
      Time for pack_array/unpack_array:     0.392/0.324 s (1.90/2.30 GB/s)) 	cr:  44.6x
      Time for pack_array2/unpack_array2:   0.199/0.127 s (3.75/5.88 GB/s)) 	cr:  44.6x
      Time for compress/decompress:         0.185/0.067 s (4.02/11.07 GB/s)) 	cr:  44.6x
    Using *** Codec.ZSTD *** compressor:
      Time for pack_array/unpack_array:     0.443/0.326 s (1.68/2.29 GB/s)) 	cr:  78.8x
      Time for pack_array2/unpack_array2:   0.259/0.108 s (2.88/6.89 GB/s)) 	cr:  78.8x
      Time for compress/decompress:         0.238/0.062 s (3.13/12.02 GB/s)) 	cr:  78.8x

    *** the random distribution ***
    Using *** Codec.BLOSCLZ *** compressor:
      Time for pack_array/unpack_array:     0.328/0.329 s (2.27/2.26 GB/s)) 	cr:   4.0x
      Time for pack_array2/unpack_array2:   0.434/0.116 s (1.72/6.40 GB/s)) 	cr:   4.0x
      Time for compress/decompress:         0.128/0.071 s (5.83/10.56 GB/s)) 	cr:   4.0x
    Using *** Codec.LZ4 *** compressor:
      Time for pack_array/unpack_array:     0.332/0.324 s (2.24/2.30 GB/s)) 	cr:   4.0x
      Time for pack_array2/unpack_array2:   0.430/0.139 s (1.73/5.37 GB/s)) 	cr:   4.0x
      Time for compress/decompress:         0.129/0.070 s (5.77/10.58 GB/s)) 	cr:   4.0x
    Using *** Codec.LZ4HC *** compressor:
      Time for pack_array/unpack_array:     0.662/0.330 s (1.13/2.26 GB/s)) 	cr:   4.0x
      Time for pack_array2/unpack_array2:   0.759/0.094 s (0.98/7.89 GB/s)) 	cr:   4.0x
      Time for compress/decompress:         0.441/0.068 s (1.69/10.95 GB/s)) 	cr:   4.0x
    Using *** Codec.ZLIB *** compressor:
      Time for pack_array/unpack_array:     0.678/0.349 s (1.10/2.14 GB/s)) 	cr:   4.7x
      Time for pack_array2/unpack_array2:   0.753/0.120 s (0.99/6.23 GB/s)) 	cr:   4.7x
      Time for compress/decompress:         0.468/0.079 s (1.59/9.43 GB/s)) 	cr:   4.7x
    Using *** Codec.ZSTD *** compressor:
      Time for pack_array/unpack_array:     0.702/0.335 s (1.06/2.22 GB/s)) 	cr:   4.4x
      Time for pack_array2/unpack_array2:   0.823/0.105 s (0.91/7.10 GB/s)) 	cr:   4.4x
      Time for compress/decompress:         0.498/0.067 s (1.50/11.18 GB/s)) 	cr:   4.4x

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
