.. _Traffic:

Traffic
=======

What crossed the wire on behalf of one remote array or one :ref:`Proxy`, counted
at the transport: every range read and every chunk, cumulative from the moment
the source was built.  Bytes are the half of the block-granularity trade that
timing does not show -- a slice that reads one block of a chunk and one that
reads the whole chunk take about the same time on a fast link and differ by the
compression ratio in traffic -- so :ref:`C2Array` and :ref:`Proxy` each carry one
under ``traffic``.  Take two readings and subtract, or
:meth:`Traffic.reset` between them.

.. currentmodule:: blosc2

.. autoclass:: Traffic
    :members:
    :member-order: groupwise
