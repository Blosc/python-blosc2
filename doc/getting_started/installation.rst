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

Source code
+++++++++++

.. code-block:: console

    git clone https://github.com/Blosc/python-blosc2/
    cd python-blosc2
    pip install .[test]   # install with test dependencies

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
