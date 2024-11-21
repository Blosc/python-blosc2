.. _NDArray:

NDArray
=======

The multidimensional data array class. This class consists of a set of useful parameters and methods that allow not only to create an array correctly, but also to being able to extract multidimensional slices from it (and much more).

.. currentmodule:: blosc2.NDArray

Methods
-------

.. autosummary::
    :toctree: autofiles/ndarray
    :nosignatures:

    __iter__
    __len__
    __getitem__
    __setitem__
    copy
    get_chunk
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
    fields
    keep_last_read
    info
    schunk
    size
    cparams
    dparams
    urlpath
    vlmeta


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
    nans
    ndarray_from_cframe
    uninit
    zeros
    ones
    full
    arange
    linspace
    reshape
