Announcing Python-Blosc2 3.0.0-beta.4
=====================================

The Blosc development team is pleased to announce the fourth beta release of
Python-Blosc2 3.0.0.  Here, documentation has been improved quite a lot and
we have added more examples and tutorials (thanks NumFOCUS for sponsoring this).
Also, there are new CParams, DParams and Storage dataclasses that allow for
a more flexible and powerful way to set parameters for the Blosc2 compressor.

In new 3.0 release, you can evaluate expressions like ``a + sin(b) + 1`` where
``a`` and ``b`` are NDArray instances.  This is a powerful feature that allows for
efficient computations on compressed data, and supports advanced features
like reductions, filters, user-defined functions and broadcasting (still
in beta).  See this
`example <https://github.com/Blosc/python-blosc2/blob/main/examples/ndarray/eval_expr.py>`_.

Also, we have added support for memory mapping in ``SChunk`` and ``NDArray`` instances.
This allows to map super-chunks stored in disk and access them as if they were in
memory.  When combined with the evaluation engine, this feature allows for very
good performance when working with large datasets.  See this
`benchmark <https://github.com/Blosc/python-blosc2/blob/main/bench/ndarray/lazyarray-expr.ipynb>`_
(as it is a Jupyter notebook, you can easily run it in your own computer).

Last, but not least, we are using NumPy 2.x as the default for testing procedures
and builds. This means that our wheels are built against NumPy 2, so in case you want
to use NumPy 1.x, you will need to use NumPy 1.23.0 or later.

As always, we would like to get feedback from the community before the final release.
We are providing binary wheels that you can easily install from PyPI with:

    pip install blosc2==3.0.0b4

For more info, you can have a look at the release notes in:

https://github.com/Blosc/python-blosc2/releases

More docs and examples are available in the documentation site:

https://www.blosc.org/python-blosc2/python-blosc2.html

What is it?
-----------

`C-Blosc2 <https://github.com/Blosc/c-blosc2>`_ is the new major version of
`C-Blosc <https://github.com/Blosc/c-blosc>`_, and is backward compatible with
both the C-Blosc1 API and its in-memory format. Python-Blosc2 is a Python
package that wraps C-Blosc2, the newest version of the Blosc compressor.

Starting with version 3.0.0, Python-Blosc2 is including a powerful computing
engine that can operate on compressed data that can be either in-memory,
on-disk or on the network. This engine also supports advanced features like
reductions, filters, user-defined functions and broadcasting.

You can read some of our tutorials on how to perform advanced computations at:

* https://github.com/Blosc/python-blosc2/blob/main/doc/getting_started/tutorials/03.lazyarray-expressions.ipynb
* https://github.com/Blosc/python-blosc2/blob/main/doc/getting_started/tutorials/03.lazyarray-udf.ipynb

In addition, Python-Blosc2 aims to leverage the full C-Blosc2 functionality to
support super-chunks
(`SChunk <https://www.blosc.org/python-blosc2/reference/schunk_api.html>`_),
multi-dimensional arrays
(`NDArray <https://www.blosc.org/python-blosc2/reference/ndarray_api.html>`_),
metadata, serialization and other bells and whistles introduced in C-Blosc2.

**Note:** Blosc2 is meant to be backward compatible with Blosc(1) data.
That means that it can read data generated with Blosc, but the opposite
is not true (i.e. there is no *forward* compatibility).

Sources repository
------------------

The sources and documentation are managed through github services at:

https://github.com/Blosc/python-blosc2

Python-Blosc2 is distributed using the BSD license, see
https://github.com/Blosc/python-blosc2/blob/main/LICENSE.txt
for details.

Mastodon feed
-------------

Please follow https://fosstodon.org/@Blosc2 to get informed about the latest
developments.


- Blosc Development Team
  Make compression better
