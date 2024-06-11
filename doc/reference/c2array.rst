.. _C2Array:

C2Array API
===========

This is a class for remote arrays. This kind of array can also work as operand on a LazyExpr, LazyUDF or reduction.

.. currentmodule:: blosc2.C2Array

Methods
-------

.. autosummary::
    :toctree: autofiles/c2array
    :nosignatures:

    __init__
    __getitem__

Attributes
----------

.. autosummary::
    :toctree: autofiles/c2array

    shape
    ext_shape
    chunks
    blocks
    dtype

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

    c2sub_urlbase
    c2sub_auth_cookie
