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
    copy
    empty
    empty_like
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
    uninit
    zeros
    zeros_like



.. autofunction:: blosc2.arange
.. autofunction:: blosc2.asarray
.. autofunction:: blosc2.copy
.. autofunction:: blosc2.empty
.. autofunction:: blosc2.empty_like
.. autofunction:: blosc2.eye
.. autofunction:: blosc2.frombuffer
.. autofunction:: blosc2.fromiter
.. autofunction:: blosc2.full
.. autofunction:: blosc2.full_like
.. autofunction:: blosc2.linspace
.. autofunction:: blosc2.meshgrid
.. autofunction:: blosc2.nans
.. autofunction:: blosc2.ndarray_from_cframe
.. autofunction:: blosc2.ones
.. autofunction:: blosc2.ones_like
.. autofunction:: blosc2.reshape
.. autofunction:: blosc2.uninit
.. autofunction:: blosc2.zeros
.. autofunction:: blosc2.zeros_like
