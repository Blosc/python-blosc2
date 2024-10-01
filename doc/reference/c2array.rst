.. _C2Array:

C2Array
=======

This is a class for remote arrays. This kind of array can also work as operand on a LazyExpr, LazyUDF or reduction.

.. currentmodule:: blosc2.C2Array

Methods
-------

.. autosummary::
    :toctree: autofiles/c2array
    :nosignatures:

    __init__
    __getitem__
    get_chunk

Attributes
----------

.. autosummary::
    :toctree: autofiles/c2array

    shape
    chunks
    blocks
    dtype
    cparams

.. _URLPath:

URLPath class
-------------

.. currentmodule:: blosc2.URLPath

.. autosummary::
    :toctree: autofiles/URLPath

    __init__

Context managers
----------------

.. currentmodule:: blosc2

.. autosummary::
    :toctree: autofiles/c2array

    c2context
