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
    empty_like
    expand_dims
    eye
    frombuffer
    fromiter
    full
    full_like
    linspace
    meshgrid
    nans
    ndarray_from_cframe
    ones
    ones_like
    reshape
    stack
    squeeze
    uninit
    zeros
    zeros_like

.. autofunction:: arange
.. autofunction:: asarray
.. autofunction:: concat
.. autofunction:: copy
.. autofunction:: empty
.. autofunction:: empty_like
.. autofunction:: expand_dims
.. autofunction:: eye
.. autofunction:: frombuffer
.. autofunction:: fromiter
.. autofunction:: full
.. autofunction:: full_like
.. autofunction:: linspace
.. autofunction:: nans
.. autofunction:: ndarray_from_cframe
.. autofunction:: ones
.. autofunction:: ones_like
.. autofunction:: reshape
.. autofunction:: stack
.. autofunction:: uninit
.. autofunction:: zeros
.. autofunction:: zeros_like
