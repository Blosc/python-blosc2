.. _Array:

Array
=====

Minimal typing protocol for array-like objects compatible with blosc2.

This protocol describes the basic interface required by blosc2 arrays.
It is implemented by blosc2 classes (:ref:`NDArray`, :ref:`NDField`,
:ref:`LazyArray`, :ref:`C2Array`, :ref:`ProxyNDSource`...)
and is compatible with NumPy arrays and other array-like containers
(e.g., PyTorch, TensorFlow, Dask, Zarr, ...).

.. currentmodule:: blosc2

.. autoclass:: Array

    :Special Methods:

    .. autosummary::

        __len__
        __getitem__
