.. _C2Array:

C2Array
=======

This is a class for remote arrays. This kind of array can also work as operand on a LazyExpr, LazyUDF or reduction.

Wrapped in a :ref:`Proxy`, a stored remote array is read at block granularity:
the proxy asks for the blocks a slice touches rather than the chunks they live
in, which for a multi-megabyte chunk is a small fraction of the bytes.  That
rests on the subscriber serving the dataset from a file, ``Range`` header and
auth cookie both honoured; a dataset it computes instead (a lazy expression, an
HDF5 leaf) is fetched a whole chunk at a time, as everything was before.  Which
one this is takes at most one request to find out, and is decided once --
:meth:`C2Array.block_source` is what answers it.


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


.. _C2NDSource:

C2NDSource class
----------------
.. autoclass:: C2NDSource
    :members:
    :member-order: groupwise


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
