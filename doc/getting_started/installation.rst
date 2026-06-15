Installation
============
You can install Python-Blosc2 wheels via PyPI using Pip, Conda or clone the GitHub repository.

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

The base install already includes everything needed for compression, the
array machinery, and the :doc:`b2view <b2view>` terminal browser.  A few
heavier, feature-specific dependencies are kept out of it and grouped
into *extras* that you opt into with the ``blosc2[extra]`` syntax:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Extra
     - Adds
   * - ``hires``
     - The high-resolution image view in b2view (the ``h`` key), which
       renders a real ``matplotlib`` image in the terminal
       (``textual-image``, ``matplotlib``).  The lightweight braille plot
       (the ``p`` key) is built in and needs no extra.
   * - ``parquet``
     - The ``parquet-to-blosc2`` converter (``pyarrow``); see
       :doc:`parquet_to_blosc2`.

Install one or more extras by listing them in brackets (quote the
argument in shells like ``zsh`` that treat brackets specially):

.. code-block:: console

    pip install "blosc2[hires]"           # b2view high-res view (h key)
    pip install "blosc2[parquet]"         # the Parquet converter
    pip install "blosc2[hires,parquet]"   # both at once

Source code
+++++++++++

.. code-block:: console

    git clone https://github.com/Blosc/python-blosc2/
    cd python-blosc2
    pip install . --group test   # install with test dependencies

That's all. You can proceed with testing section now.

Testing
-------

After compiling, you can quickly check that the package is sane by
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
