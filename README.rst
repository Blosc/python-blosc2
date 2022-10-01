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

     PYTHONPATH=. python bench/pack_compress.py

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
      Time for copying array with np.copy:                  0.394 s (3.79 GB/s))


    *** the arange linear distribution ***
    Using *** Codec.BLOSCLZ *** compressor:
      Time for compress/decompress:         0.051/0.101 s (29.08/14.80 GB/s)) 	cr: 444.3x
      Time for pack_array/unpack_array:     0.600/0.764 s (2.49/1.95 GB/s)) 	cr: 442.3x
      Time for pack_array2/unpack_array2:   0.059/0.158 s (25.28/9.44 GB/s)) 	cr: 444.2x
    Using *** Codec.LZ4 *** compressor:
      Time for compress/decompress:         0.059/0.116 s (25.07/12.82 GB/s)) 	cr: 279.2x
      Time for pack_array/unpack_array:     0.615/0.758 s (2.42/1.97 GB/s)) 	cr: 277.9x
      Time for pack_array2/unpack_array2:   0.058/0.160 s (25.52/9.31 GB/s)) 	cr: 279.2x
    Using *** Codec.LZ4HC *** compressor:
      Time for compress/decompress:         0.193/0.085 s (7.71/17.45 GB/s)) 	cr: 155.9x
      Time for pack_array/unpack_array:     0.786/0.754 s (1.89/1.98 GB/s)) 	cr: 155.4x
      Time for pack_array2/unpack_array2:   0.218/0.165 s (6.84/9.02 GB/s)) 	cr: 155.9x
    Using *** Codec.ZLIB *** compressor:
      Time for compress/decompress:         0.250/0.141 s (5.96/10.55 GB/s)) 	cr: 273.8x
      Time for pack_array/unpack_array:     0.799/0.845 s (1.87/1.76 GB/s)) 	cr: 273.2x
      Time for pack_array2/unpack_array2:   0.261/0.243 s (5.71/6.13 GB/s)) 	cr: 273.8x
    Using *** Codec.ZSTD *** compressor:
      Time for compress/decompress:         0.189/0.079 s (7.89/18.92 GB/s)) 	cr: 644.9x
      Time for pack_array/unpack_array:     0.725/0.770 s (2.06/1.94 GB/s)) 	cr: 630.9x
      Time for pack_array2/unpack_array2:   0.206/0.143 s (7.25/10.39 GB/s)) 	cr: 644.8x

    *** the linspace linear distribution ***
    Using *** Codec.BLOSCLZ *** compressor:
      Time for compress/decompress:         0.091/0.113 s (16.34/13.21 GB/s)) 	cr:  50.1x
      Time for pack_array/unpack_array:     0.623/0.751 s (2.39/1.98 GB/s)) 	cr:  50.0x
      Time for pack_array2/unpack_array2:   0.124/0.163 s (11.98/9.12 GB/s)) 	cr:  50.1x
    Using *** Codec.LZ4 *** compressor:
      Time for compress/decompress:         0.077/0.114 s (19.33/13.12 GB/s)) 	cr:  55.7x
      Time for pack_array/unpack_array:     0.624/0.740 s (2.39/2.01 GB/s)) 	cr:  55.8x
      Time for pack_array2/unpack_array2:   0.098/0.190 s (15.19/7.83 GB/s)) 	cr:  55.7x
    Using *** Codec.LZ4HC *** compressor:
      Time for compress/decompress:         0.352/0.075 s (4.23/19.98 GB/s)) 	cr:  53.6x
      Time for pack_array/unpack_array:     0.918/0.781 s (1.62/1.91 GB/s)) 	cr:  53.6x
      Time for pack_array2/unpack_array2:   0.389/0.139 s (3.83/10.72 GB/s)) 	cr:  53.6x
    Using *** Codec.ZLIB *** compressor:
      Time for compress/decompress:         0.395/0.148 s (3.77/10.08 GB/s)) 	cr:  50.4x
      Time for pack_array/unpack_array:     0.940/0.824 s (1.59/1.81 GB/s)) 	cr:  50.4x
      Time for pack_array2/unpack_array2:   0.433/0.252 s (3.44/5.92 GB/s)) 	cr:  50.4x
    Using *** Codec.ZSTD *** compressor:
      Time for compress/decompress:         0.402/0.098 s (3.71/15.22 GB/s)) 	cr:  74.7x
      Time for pack_array/unpack_array:     0.949/0.782 s (1.57/1.91 GB/s)) 	cr:  74.7x
      Time for pack_array2/unpack_array2:   0.426/0.175 s (3.50/8.49 GB/s)) 	cr:  74.7x

    *** the random distribution ***
    Using *** Codec.BLOSCLZ *** compressor:
      Time for compress/decompress:         0.240/0.119 s (6.22/12.48 GB/s)) 	cr:   4.0x
      Time for pack_array/unpack_array:     0.794/0.767 s (1.88/1.94 GB/s)) 	cr:   4.0x
      Time for pack_array2/unpack_array2:   0.578/0.162 s (2.58/9.20 GB/s)) 	cr:   4.0x
    Using *** Codec.LZ4 *** compressor:
      Time for compress/decompress:         0.250/0.114 s (5.97/13.11 GB/s)) 	cr:   4.0x
      Time for pack_array/unpack_array:     0.794/0.767 s (1.88/1.94 GB/s)) 	cr:   4.0x
      Time for pack_array2/unpack_array2:   0.590/0.161 s (2.53/9.24 GB/s)) 	cr:   4.0x
    Using *** Codec.LZ4HC *** compressor:
      Time for compress/decompress:         1.102/0.088 s (1.35/17.01 GB/s)) 	cr:   4.0x
      Time for pack_array/unpack_array:     1.690/0.758 s (0.88/1.97 GB/s)) 	cr:   4.0x
      Time for pack_array2/unpack_array2:   1.445/0.178 s (1.03/8.38 GB/s)) 	cr:   4.0x
    Using *** Codec.ZLIB *** compressor:
      Time for compress/decompress:         1.258/0.210 s (1.18/7.11 GB/s)) 	cr:   4.7x
      Time for pack_array/unpack_array:     1.822/0.898 s (0.82/1.66 GB/s)) 	cr:   4.7x
      Time for pack_array2/unpack_array2:   1.549/0.355 s (0.96/4.20 GB/s)) 	cr:   4.7x
    Using *** Codec.ZSTD *** compressor:
      Time for compress/decompress:         1.653/0.098 s (0.90/15.21 GB/s)) 	cr:   4.4x
      Time for pack_array/unpack_array:     2.206/0.796 s (0.68/1.87 GB/s)) 	cr:   4.4x
      Time for pack_array2/unpack_array2:   2.077/0.179 s (0.72/8.30 GB/s)) 	cr:   4.4x

As can be seen, is perfectly possible for python-blosc2 to go faster than a plain memcpy(). But more interestingly, you can easily choose the codecs and filters that better adapt to your datasets, and persist and transmit them faster and using less memory.

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
