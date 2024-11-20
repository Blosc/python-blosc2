.. _NDField:

NDField
=======

This class is used to represent fields of a structured :ref:`NDArray <NDArray>`.

For instance, you can create an array with two fields::

    s = blosc2.empty(shape, dtype=[("a", np.float32), ("b", np.float64)])
    a = blosc2.NDField(s, "a")
    b = blosc2.NDField(s, "b")

.. currentmodule:: blosc2.NDField

Methods
-------

.. autosummary::
    :toctree: autofiles/ndfield
    :nosignatures:

    __init__
    __iter__
    __len__
    __getitem__
    __setitem__

Attributes
----------

.. autosummary::
    :toctree: autofiles/ndfield

    schunk
    shape
