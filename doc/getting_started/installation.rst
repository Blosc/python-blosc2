Installation
============
You can install binary Python-Blosc2 wheels from PyPI with pip, from conda-forge with conda, or build from a clone of the GitHub repository.

Pip
+++

.. code-block::

    pip install blosc2 --upgrade

Conda
+++++

.. code-block::

    conda install -c conda-forge python-blosc2

Optional features (extras)
++++++++++++++++++++++++++

The base install includes everything needed for compression and the array
machinery.  Heavier, feature-specific dependencies are kept out of it and
grouped into *extras* that you opt into with the ``blosc2[extra]`` syntax:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Extra
     - Adds
   * - ``tui``
     - The :doc:`b2view <../guides/b2view>` terminal browser (``textual``,
       ``textual-plotext``), including its in-terminal braille plot (the
       ``p`` key).  Required by the ``b2view`` command.
   * - ``hires``
     - The high-resolution image view in b2view (the ``h`` key), which
       renders a real ``matplotlib`` image in the terminal
       (``textual-image``, ``matplotlib``).  Includes ``tui``.
   * - ``parquet``
     - The ``parquet-to-blosc2`` converter (``pyarrow``); see
       :doc:`../guides/parquet_to_blosc2`.
   * - ``fsspec``
     - Reading and writing single-file containers through any `fsspec
       <https://filesystem-spec.readthedocs.io>`_ URL.  The driver for each
       protocol is a separate install (``s3fs`` for ``s3://``, ``gcsfs`` for
       ``gs://``, ``adlfs`` for ``abfs://``...), and credentials are configured
       through the driver, not through blosc2.

Install one or more extras by listing them in brackets (quote the
argument in shells like ``zsh`` that treat brackets specially):

.. code-block:: console

    pip install "blosc2[tui]"             # the b2view terminal browser
    pip install "blosc2[hires]"           # b2view + its high-res view (h key)
    pip install "blosc2[parquet]"         # the Parquet converter
    pip install "blosc2[fsspec]" s3fs     # fsspec URLs, plus the S3 driver
    pip install "blosc2[tui,parquet]"     # several at once

With the ``fsspec`` extra, :func:`blosc2.open` and :func:`blosc2.save_array`
accept any fsspec URL, including chained ones::

    blosc2.open("s3://bucket/array.b2nd")
    blosc2.open("zip://inner.b2nd::s3://bucket/archive.zip")

The whole object is transferred in one go, so this covers single-file
containers (``.b2nd``, ``.b2f``, ``.b2e``, ``.b2z``) in read mode.  Passing a
cache directory downloads the container instead and opens it locally, which
additionally covers directory containers (``.b2d`` stores, sparse frames),
``offset`` and ``mmap_mode``, and makes repeated opens cheap::

    blosc2.open("s3://bucket/store.b2d", cache_storage="~/.cache/blosc2")

There is no default cache directory on purpose, so nothing writes to your disk
unless you name the place.

For a container too big to transfer at all, ``lazy=True`` leaves the frame where
it is and reads only the chunks a slice touches, one range request each::

    a = blosc2.open("s3://bucket/huge.b2nd", lazy=True)
    a[1000:1010]   # fetches one or two chunks, not the array

This returns a :ref:`Proxy` over the remote frame, so what it fetched stays
cached in it.  It needs a contiguous frame holding an :ref:`NDArray`.

Source code
+++++++++++

.. code-block:: console

    git clone https://github.com/Blosc/python-blosc2/
    cd python-blosc2
    pip install . --group test   # install with test dependencies

(the ``--group`` flag needs pip >= 25.1). That's all. You can proceed
with the testing section now.

Testing
-------

After installing, you can quickly check that the package is sane by
running the tests:

.. code-block:: console

    pytest  # add -v for verbose mode

Benchmarking
------------

If curious, you may want to run a small benchmark that compares a plain
NumPy array copy against compression through different compressors in
your Blosc build:

.. code-block:: console

     PYTHONPATH=. python bench/pack_compress.py
