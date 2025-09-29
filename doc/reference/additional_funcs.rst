Additional Functions and Type Utilities
=======================================

Functions
---------

The following functions can also be used for computing with any of :ref:`NDArray <NDArray>`, :ref:`C2Array <C2Array>`, :ref:`NDField <NDField>` and :ref:`LazyExpr <LazyExpr>`.

Their result is typically a :ref:`LazyExpr` instance, which can be evaluated (with ``compute`` or ``__getitem__``) to get the actual values of the computation.

.. currentmodule:: blosc2

.. autosummary::

    clip
    conj
    contains
    imag
    real
    round
    where



.. autofunction:: blosc2.clip
.. autofunction:: blosc2.conj
.. autofunction:: blosc2.contains
.. autofunction:: blosc2.imag
.. autofunction:: blosc2.real
.. autofunction:: blosc2.round
.. autofunction:: blosc2.where
Type Utilities
--------------

The following functions are useful for working with datatypes.

.. currentmodule:: blosc2

.. autosummary::

    astype
    can_cast
    isdtype
    result_type



.. autofunction:: blosc2.astype
.. autofunction:: blosc2.can_cast
.. autofunction:: blosc2.isdtype
.. autofunction:: blosc2.result_type
