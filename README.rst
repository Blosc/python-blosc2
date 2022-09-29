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

Just to whet your appetite, here are some speed figures for an Intel box (i9-10940X CPU @ 3.30GHz, 14 cores)
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
    Platform: Linux-5.15.0-41-generic-x86_64 (#44-Ubuntu SMP Wed Jun 22 14:20:53 UTC 2022)
    Linux dist: Ubuntu 22.04 LTS
    Processor: x86_64
    Byte-ordering: little
    Detected cores: 14.0
    Number of threads to use by default: 8
    -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    Creating NumPy arrays with 10**8 int64/float64 elements:
      Time for copying array with np.copy:                  0.196 s (3.80 GB/s))

    *** the arange linear distribution ***
    Using *** Codec.BLOSCLZ *** compressor:
      Time for pack_array/unpack_array:     0.312/0.414 s (2.39/1.80 GB/s)) 	cr: 441.6x
      Time for pack_array2/unpack_array2:   0.039/0.084 s (19.02/8.88 GB/s)) 	cr: 444.0x
      Time for compress/decompress:         0.037/0.057 s (20.26/12.98 GB/s)) 	cr: 444.1x
    Using *** Codec.LZ4 *** compressor:
      Time for pack_array/unpack_array:     0.308/0.384 s (2.42/1.94 GB/s)) 	cr: 277.7x
      Time for pack_array2/unpack_array2:   0.037/0.096 s (20.27/7.80 GB/s)) 	cr: 279.2x
      Time for compress/decompress:         0.034/0.053 s (22.19/13.98 GB/s)) 	cr: 279.2x
    Using *** Codec.LZ4HC *** compressor:
      Time for pack_array/unpack_array:     0.423/0.386 s (1.76/1.93 GB/s)) 	cr: 155.4x
      Time for pack_array2/unpack_array2:   0.119/0.094 s (6.27/7.94 GB/s)) 	cr: 155.9x
      Time for compress/decompress:         0.120/0.044 s (6.21/16.77 GB/s)) 	cr: 155.9x
    Using *** Codec.ZLIB *** compressor:
      Time for pack_array/unpack_array:     0.404/0.423 s (1.84/1.76 GB/s)) 	cr: 273.3x
      Time for pack_array2/unpack_array2:   0.139/0.126 s (5.38/5.90 GB/s)) 	cr: 273.8x
      Time for compress/decompress:         0.130/0.078 s (5.75/9.58 GB/s)) 	cr: 273.8x
    Using *** Codec.ZSTD *** compressor:
      Time for pack_array/unpack_array:     0.398/0.410 s (1.87/1.82 GB/s)) 	cr: 630.8x
      Time for pack_array2/unpack_array2:   0.121/0.088 s (6.16/8.50 GB/s)) 	cr: 644.7x
      Time for compress/decompress:         0.112/0.045 s (6.65/16.58 GB/s)) 	cr: 644.9x

    *** the linspace linear distribution ***
    Using *** Codec.BLOSCLZ *** compressor:
      Time for pack_array/unpack_array:     0.333/0.398 s (2.24/1.87 GB/s)) 	cr:  35.7x
      Time for pack_array2/unpack_array2:   0.095/0.096 s (7.81/7.74 GB/s)) 	cr:  35.6x
      Time for compress/decompress:         0.076/0.062 s (9.82/12.02 GB/s)) 	cr:  35.6x
    Using *** Codec.LZ4 *** compressor:
      Time for pack_array/unpack_array:     0.327/0.398 s (2.28/1.87 GB/s)) 	cr:  40.5x
      Time for pack_array2/unpack_array2:   0.063/0.095 s (11.91/7.82 GB/s)) 	cr:  40.5x
      Time for compress/decompress:         0.059/0.060 s (12.73/12.45 GB/s)) 	cr:  40.5x
    Using *** Codec.LZ4HC *** compressor:
      Time for pack_array/unpack_array:     0.555/0.406 s (1.34/1.83 GB/s)) 	cr:  44.7x
      Time for pack_array2/unpack_array2:   0.291/0.093 s (2.56/8.04 GB/s)) 	cr:  44.7x
      Time for compress/decompress:         0.259/0.036 s (2.88/20.49 GB/s)) 	cr:  44.7x
    Using *** Codec.ZLIB *** compressor:
      Time for pack_array/unpack_array:     0.516/0.427 s (1.44/1.74 GB/s)) 	cr:  44.6x
      Time for pack_array2/unpack_array2:   0.265/0.132 s (2.82/5.67 GB/s)) 	cr:  44.6x
      Time for compress/decompress:         0.235/0.060 s (3.17/12.33 GB/s)) 	cr:  44.6x
    Using *** Codec.ZSTD *** compressor:
      Time for pack_array/unpack_array:     0.470/0.396 s (1.58/1.88 GB/s)) 	cr:  78.8x
      Time for pack_array2/unpack_array2:   0.189/0.099 s (3.93/7.53 GB/s)) 	cr:  78.8x
      Time for compress/decompress:         0.183/0.072 s (4.07/10.36 GB/s)) 	cr:  78.8x

    *** the random distribution ***
    Using *** Codec.BLOSCLZ *** compressor:
      Time for pack_array/unpack_array:     0.419/0.401 s (1.78/1.86 GB/s)) 	cr:   4.0x
      Time for pack_array2/unpack_array2:   0.301/0.076 s (2.48/9.85 GB/s)) 	cr:   4.0x
      Time for compress/decompress:         0.148/0.059 s (5.03/12.70 GB/s)) 	cr:   4.0x
    Using *** Codec.LZ4 *** compressor:
      Time for pack_array/unpack_array:     0.402/0.401 s (1.85/1.86 GB/s)) 	cr:   4.0x
      Time for pack_array2/unpack_array2:   0.310/0.090 s (2.40/8.31 GB/s)) 	cr:   4.0x
      Time for compress/decompress:         0.130/0.060 s (5.73/12.35 GB/s)) 	cr:   4.0x
    Using *** Codec.LZ4HC *** compressor:
      Time for pack_array/unpack_array:     0.866/0.411 s (0.86/1.81 GB/s)) 	cr:   4.0x
      Time for pack_array2/unpack_array2:   0.744/0.076 s (1.00/9.76 GB/s)) 	cr:   4.0x
      Time for compress/decompress:         0.568/0.062 s (1.31/12.05 GB/s)) 	cr:   4.0x
    Using *** Codec.ZLIB *** compressor:
      Time for pack_array/unpack_array:     0.961/0.446 s (0.78/1.67 GB/s)) 	cr:   4.7x
      Time for pack_array2/unpack_array2:   0.826/0.166 s (0.90/4.50 GB/s)) 	cr:   4.7x
      Time for compress/decompress:         0.681/0.107 s (1.09/6.96 GB/s)) 	cr:   4.7x
    Using *** Codec.ZSTD *** compressor:
      Time for pack_array/unpack_array:     1.105/0.414 s (0.67/1.80 GB/s)) 	cr:   4.4x
      Time for pack_array2/unpack_array2:   1.066/0.093 s (0.70/7.99 GB/s)) 	cr:   4.4x
      Time for compress/decompress:         0.828/0.052 s (0.90/14.45 GB/s)) 	cr:   4.4x

As can be seen, is perfectly possible for python-blosc2 to go faster than a plain memcpy(). But more interestingly, you can persist and transmit data faster and using less memory.

Start using compression in your data workflows and feel the experience of doing more with less.

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
