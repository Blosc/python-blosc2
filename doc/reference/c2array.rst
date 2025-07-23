.. _C2Array:

C2Array
=======

This is a class for remote arrays. This kind of array can also work as operand on a LazyExpr, LazyUDF or reduction.


.. currentmodule:: blosc2

.. autoclass:: C2Array
    :members:
    :exclude-members: all, any, max, mean, min, prod, std, sum, var
    :member-order: groupwise

    :Special Methods:

    .. autosummary::
        __init__
        __getitem__

    Constructor
    -----------
    .. automethod:: __init__

    Utility Methods
    ---------------
    .. automethod:: __getitem__


.. _URLPath:

URLPath class
-------------
.. autoclass:: URLPath
    :members:
    :member-order: groupwise

    .. autosummary::
        __init__

    .. automethod:: __init__

Context managers
----------------
.. autofunction:: c2context
