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

     $ PYTHONPATH=. python bench/pack_compress.py

Just to whet your appetite, here are the results for an Apple M1 (2021)
with 8 GB of RAM but YMMV (and will vary!)::

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
    Creating a large NumPy array with 10**8 int64 elements:
      [0.00000000e+00 1.00000001e-04 2.00000002e-04 ... 9.99999980e+03
     9.99999990e+03 1.00000000e+04]
      Time for copying array with np.copy:	        		0.167 s (4.46 GB/s))
      Time for copying array with np.copyto and empty_like:	0.038 s (19.77 GB/s))
      Time for copying array with np.copyto and zeros:  	0.039 s (19.16 GB/s))
      Time for copying array with np.copyto and full_like:	0.032 s (23.40 GB/s))
      Time for copying array with numpy assignment:		    0.032 s (23.08 GB/s))

    Using *** blosclz *** compressor:
      Time for pack_array/unpack_array:     0.157/0.244 s (4.75/3.05 GB/s)).	Compr ratio: 27.02
      Time for compress/decompress:         0.080/0.028 s (9.28/26.82 GB/s)).	Compr ratio: 27.02
    Using *** lz4 *** compressor:
      Time for pack_array/unpack_array:     0.107/0.186 s (6.95/4.00 GB/s)).	Compr ratio: 33.93
      Time for compress/decompress:         0.036/0.025 s (20.43/30.27 GB/s)).	Compr ratio: 33.95
    Using *** lz4hc *** compressor:
      Time for pack_array/unpack_array:     0.328/0.149 s (2.27/5.01 GB/s)).	Compr ratio: 26.94
      Time for compress/decompress:         0.270/0.027 s (2.76/27.28 GB/s)).	Compr ratio: 26.94
    Using *** zlib *** compressor:
      Time for pack_array/unpack_array:     0.572/0.250 s (1.30/2.98 GB/s)).	Compr ratio: 28.17
      Time for compress/decompress:         0.497/0.089 s (1.50/8.34 GB/s)).	Compr ratio: 28.17
    Using *** zstd *** compressor:
      Time for pack_array/unpack_array:     0.530/0.187 s (1.41/3.99 GB/s)).	Compr ratio: 48.57
      Time for compress/decompress:         0.437/0.046 s (1.70/16.10 GB/s)).	Compr ratio: 47.39

For matter of comparison, here it is the output for an Apple Mac Mini (2018) 3,2 GHz 6-Core i7
with 32 GB of RAM::

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
    Python version: 3.9.4 | packaged by conda-forge | (default, May 10 2021, 22:13:15)
    [Clang 11.1.0 ]
    Platform: Darwin-20.4.0-x86_64 (Darwin Kernel Version 20.4.0: Thu Apr 22 21:46:47 PDT 2021; root:xnu-7195.101.2~1/RELEASE_X86_64)
    Processor: i386
    Byte-ordering: little
    Detected cores: 12
    Number of threads to use by default: 8
    -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    Creating a large NumPy array with 10**8 int64 elements:
      [0.00000000e+00 1.00000001e-04 2.00000002e-04 ... 9.99999980e+03
     9.99999990e+03 1.00000000e+04]
      Time for copying array with np.copy:			        0.326 s (2.29 GB/s))
      Time for copying array with np.copyto and empty_like:	0.133 s (5.61 GB/s))
      Time for copying array with np.copyto and zeros:	    0.129 s (5.77 GB/s))
      Time for copying array with np.copyto and full_like:	0.076 s (9.75 GB/s))
      Time for copying array with numpy assignment:		    0.073 s (10.27 GB/s))

    Using *** blosclz *** compressor:
      Time for pack_array/unpack_array:     0.478/0.507 s (1.56/1.47 GB/s)).	Compr ratio: 27.02
      Time for compress/decompress:         0.143/0.094 s (5.22/7.94 GB/s)).	Compr ratio: 27.02
    Using *** lz4 *** compressor:
      Time for pack_array/unpack_array:     0.480/0.520 s (1.55/1.43 GB/s)).	Compr ratio: 33.93
      Time for compress/decompress:         0.136/0.091 s (5.49/8.15 GB/s)).	Compr ratio: 33.95
    Using *** lz4hc *** compressor:
      Time for pack_array/unpack_array:     0.691/0.514 s (1.08/1.45 GB/s)).	Compr ratio: 26.94
      Time for compress/decompress:         0.359/0.091 s (2.08/8.15 GB/s)).	Compr ratio: 26.94
    Using *** zlib *** compressor:
      Time for pack_array/unpack_array:     0.801/0.584 s (0.93/1.27 GB/s)).	Compr ratio: 28.17
      Time for compress/decompress:         0.470/0.165 s (1.59/4.50 GB/s)).	Compr ratio: 28.17
    Using *** zstd *** compressor:
      Time for pack_array/unpack_array:     1.078/0.543 s (0.69/1.37 GB/s)).	Compr ratio: 48.57
      Time for compress/decompress:         0.708/0.121 s (1.05/6.17 GB/s)).	Compr ratio: 47.39

Using compression becomes more sexy when using newer processors indeed.
In case you find your own results interesting, please report them back
to the authors!

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
