.. _C2Array:

C2Array
=======

This is a class for one array-like dataset addressed through a Caterva2 server.
The dataset may be a standalone ``.b2nd`` array, an HDF5 dataset, an NDArray
leaf inside a ``.b2z`` store, or a lazy/computed array.  A ``C2Array`` does not
represent or navigate a whole remote ``TreeStore`` or ``DictStore``; use
Caterva2 to select a leaf and open that leaf's path.  This kind of array can also
work as an operand on a LazyExpr, LazyUDF or reduction.  :ref:`URLPath` is
Caterva2-only, including when its ``urlbase`` is omitted and inherited from
:func:`blosc2.c2context`.

For a comparison with byte-oriented fsspec access, see
:doc:`Working with Remote Arrays <../guides/remote_arrays>`.

Wrapped in a :ref:`Proxy`, a stored remote array is read at block granularity:
the proxy asks for the blocks a slice touches rather than the chunks they live
in, which for a multi-megabyte chunk is a small fraction of the bytes.  That
rests on the server serving the dataset from a file, ``Range`` header and
auth cookie both honoured; a dataset it computes instead (a lazy expression, an
HDF5 leaf, or a ``.b2z`` member) is fetched a whole chunk at a time, as
everything was before.  Which one this is takes at most one request to find out,
and is decided once -- :meth:`C2Array.block_source` is what answers it.

A stored remote array can also be *filled*, by as many writers at once as it has
chunks.  The array is laid out first -- ``blosc2.uninit`` writes a couple of
hundred bytes whatever its size -- and then each writer posts the chunks it owns
with :meth:`C2Array.update_chunk`.  A slot nothing was written to is free, and a
write claims it; a second write to the same slot raises
:class:`blosc2.ChunkAlreadyWritten`, so two writers that both believe they own a
chunk are resolved by the array rather than by anything either of them holds.
:meth:`C2Array.written_chunks` reads how far the fill has got out of the frame's
own offsets, which is a couple of range reads and no endpoint of its own.


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


.. autoclass:: ChunkAlreadyWritten


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
