=============
Python-Blosc2
=============

A Python wrapper for the extremely fast Blosc2 compression library
==================================================================

:Author: The Blosc development team
:Contact: blosc@blosc.org
:Github: https://github.com/Blosc/python-blosc2
:Actions: |actions|
:PyPi: |version|
:NumFOCUS: |numfocus|
:Code of Conduct: |Contributor Covenant|

.. |version| image:: https://img.shields.io/pypi/v/blosc2.png
        :target: https://pypi.python.org/pypi/blosc
.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg
        :target: https://github.com/Blosc/community/blob/master/code_of_conduct.md
.. |numfocus| image:: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
        :target: https://numfocus.org
.. |actions| image:: https://github.com/Blosc/python-blosc2/actions/workflows/build.yml/badge.svg
        :target: https://github.com/Blosc/python-blosc2/actions/workflows/build.yml


What it is
==========

`C-Blosc2 <https://github.com/Blosc/c-blosc2>`_ is the new major version of
`C-Blosc <https://github.com/Blosc/c-blosc>`_, and is backward compatible with
both the C-Blosc1 API and its in-memory format.

Python-Blosc2 is a Python package that wraps C-Blosc2, the newest version of
the Blosc compressor.  Currently Python-Blosc2 already reproduces the API of
`Python-Blosc <https://github.com/Blosc/python-blosc>`_, so it can be
used as a drop-in replacement.  However, there are a `few exceptions
for a full compatibility.
<https://github.com/Blosc/python-blosc2/blob/main/RELEASE_NOTES.md#changes-from-python-blosc-to-python-blosc2>`_

In addition, Python-Blosc2 aims to leverage the new C-Blosc2 API so as to support
super-chunks, multi-dimensional arrays
(`NDArray <https://www.blosc.org/python-blosc2/reference/ndarray_api.html>`_),
serialization and other bells and whistles introduced in C-Blosc2.  Although
this is always and endless process, we are have almost catched up with the full
C-Blosc2 potential (with the convenience of Python :-).

**Note:** Python-Blosc2 is meant to be backward compatible with Python-Blosc data.
That means that it can read data generated with Python-Blosc, but the opposite
is not true (i.e. there is no *forward* compatibility).

NDArray: an N-Dimensional store
===============================

One of the latest and more exciting additions in Python-Blosc2 is the
`NDArray <https://www.blosc.org/python-blosc2/reference/ndarray_api.html>`_ object.
It can write and read n-dimensional datasets in an extremely efficient way thanks
to a n-dim 2-level partitioning, allowing to slice and dice arbitrary large and
compressed data in a more fine-grained way:

.. image:: ./images/b2nd-2level-parts.png
  :width: 100%

To wet you appetite, here it is how the `NDArray` object performs on getting slices
orthogonal to the different axis of a 4-dim dataset:

.. image:: ./images/Read-Partial-Slices-B2ND.png
  :width: 100%

We have blogged about this: https://www.blosc.org/posts/blosc2-ndim-intro


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


Building from sources
=====================

`python-blosc2` comes with the C-Blosc2 sources with it and can be built in-place:

.. code-block:: console

    git clone https://github.com/Blosc/python-blosc2/
    cd python-blosc2
    git submodule update --init --recursive
    python -m pip install -r requirements-build.txt
    python setup.py build_ext --inplace

That's all. You can proceed with testing section now.

Testing
=======

After compiling, you can quickly check that the package is sane by
running the tests:

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


License
=======

The software is licenses under a 3-Clause BSD license. A copy of the
python-blosc2 license can be found in `LICENSE.txt <https://github.com/Blosc/python-blosc2/tree/main/LICENSE.txt>`_.

Mailing list
============

Discussion about this module is welcome in the Blosc list:

blosc@googlegroups.com

https://groups.google.es/group/blosc

Twitter
=======

Please follow `@Blosc2 <https://twitter.com/Blosc2>`_ to get informed about the latest developments.

----

  **Enjoy data!**
