.. _NDArray:

NDArray API
===============

The multidimensional data array class. This class consists of a set of useful parameters and methods that allow not only to define an array correctly, but also to handle it in a simple way, being able to extract multidimensional slices from it.

.. currentmodule:: blosc2.NDArray

Methods
-------

.. autosummary::
    :toctree: autofiles/ndarray
    :nosignatures:

    __getitem__
    __setitem__
    copy
    iterchunks_info
    slice
    squeeze
    resize
    tobytes
    to_cframe

Attributes
----------

.. autosummary::
    :toctree: autofiles/ndarray

    ndim
    shape
    ext_shape
    chunks
    ext_chunks
    blocks
    blocksize
    chunksize
    dtype
    info
    schunk
    size


.. currentmodule:: blosc2

Constructors
------------

.. autosummary::
    :toctree: autofiles/ndarray
    :nosignatures:

    asarray
    copy
    empty
    frombuffer
    full
    ndarray_from_cframe
    uninit
    zeros

Functions
---------

.. autosummary::
    :toctree: autofiles/ndarray
    :nosignatures:

    sin
    cos
    tan
    sinh
    cosh
    tanh
    arcsin
    arccos
    arctan
    arctan2
    arcsinh
    arccosh
    arctanh
    exp
    expm1
    log
    log10
    log1p
    sqrt
    conj
    real
    imag
    contains
    abs
