.. _NDArray:

NDArray
=======

The multidimensional data array class.

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
    indices
    iterchunks_info
    reshape
    resize
    save
    slice
    sort
    squeeze
    tobytes
    to_cframe

In addition, all the functions from the :ref:`Lazy Functions <lazy_functions>` section can be used with NDArray instances.


Attributes
----------

.. autosummary::
    :toctree: autofiles/ndarray

    T
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
    concat
    empty
    expand_dims
    frombuffer
    fromiter
    nans
    ndarray_from_cframe
    uninit
    zeros
    ones
    full
    arange
    linspace
    eye
    reshape
    stack
