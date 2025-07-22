.. _NDArray:

NDArray
=======

The multidimensional data array class. Instances may be constructed using the constructor functions in the list below `NDArrayConstructors`_.
In addition, all the functions from the :ref:`Lazy Functions <lazy_functions>` section can be used with NDArray instances.

.. currentmodule:: blosc2

.. autoclass:: NDArray
    :members:
    :inherited-members:
    :exclude-members: get_slice, set_slice, get_slice_numpy, get_oindex_numpy, set_oindex_numpy
    :member-order: groupwise

    :Special Methods:

    .. autosummary::

        __iter__
        __len__
        __getitem__
        __setitem__

    Utility Methods
    ---------------

    .. automethod:: __iter__
    .. automethod:: __len__
    .. automethod:: __getitem__
    .. automethod:: __setitem__

Constructors
------------
.. _NDArrayConstructors:
.. autosummary::

    arange
    asarray
    concat
    copy
    empty
    expand_dims
    eye
    frombuffer
    fromiter
    full
    linspace
    nans
    ndarray_from_cframe
    ones
    reshape
    stack
    uninit
    zeros

.. autofunction:: arange
.. autofunction:: asarray
.. autofunction:: concat
.. autofunction:: copy
.. autofunction:: empty
.. autofunction:: expand_dims
.. autofunction:: eye
.. autofunction:: frombuffer
.. autofunction:: fromiter
.. autofunction:: full
.. autofunction:: linspace
.. autofunction:: nans
.. autofunction:: ndarray_from_cframe
.. autofunction:: ones
.. autofunction:: reshape
.. autofunction:: stack
.. autofunction:: uninit
.. autofunction:: zeros
