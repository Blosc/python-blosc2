.. _NDField:

NDField
=======

This class is used to represent fields of a structured :ref:`NDArray <NDArray>`.

For instance, you can create an array with two fields::

    s = blosc2.empty(shape, dtype=[("a", np.float32), ("b", np.float64)])
    a = blosc2.NDField(s, "a")
    b = blosc2.NDField(s, "b")

.. currentmodule:: blosc2

.. autoclass:: NDField
    :members:
    :exclude-members: all, any, max, mean, min, prod, std, sum, var
    :member-order: groupwise

    :Special Methods:

    .. autosummary::

        __init__
        __iter__
        __len__
        __getitem__
        __setitem__

    Constructor
    -----------
    .. automethod:: __init__

    Utility Methods
    ---------------
    .. automethod:: __iter__
    .. automethod:: __len__
    .. automethod:: __getitem__
    .. automethod:: __setitem__
