Universal Functions (`ufuncs`)
------------------------------

The following elementwise functions can be used for computing with any of :ref:`NDArray <NDArray>`, :ref:`C2Array <C2Array>`, :ref:`NDField <NDField>` and :ref:`LazyExpr <LazyExpr>`.

Their result is always a :ref:`LazyExpr` instance, which can be evaluated (with ``compute`` or ``__getitem__``) to get the actual values of the computation.

Note: The functions ``conj``, ``real``, ``imag``, ``contains``, ``where`` are not technically ufuncs.

.. currentmodule:: blosc2

.. autosummary::

    abs
    acos
    acosh
    add
    arccos
    arccosh
    arcsin
    arcsinh
    arctan
    arctan2
    arctanh
    asin
    asinh
    atan
    atan2
    atanh
    bitwise_and
    bitwise_invert
    bitwise_left_shift
    bitwise_or
    bitwise_right_shift
    bitwise_xor
    ceil
    conj
    copysign
    cos
    cosh
    divide
    equal
    exp
    expm1
    floor
    floor_divide
    greater
    greater_equal
    hypot
    isfinite
    isinf
    isnan
    less
    less_equal
    log
    log1p
    log2
    log10
    logaddexp
    logical_and
    logical_not
    logical_or
    logical_xor
    matmul
    maximum
    minimum
    multiply
    negative
    nextafter
    not_equal
    positive
    pow
    reciprocal
    remainder
    sign
    signbit
    sin
    sinh
    sqrt
    square
    subtract
    tan
    tanh
    trunc
    vecdot



.. autofunction:: blosc2.abs
.. autofunction:: blosc2.acos
.. autofunction:: blosc2.acosh
.. autofunction:: blosc2.add
.. autofunction:: blosc2.arccos
.. autofunction:: blosc2.arccosh
.. autofunction:: blosc2.arcsin
.. autofunction:: blosc2.arcsinh
.. autofunction:: blosc2.arctan
.. autofunction:: blosc2.arctan2
.. autofunction:: blosc2.arctanh
.. autofunction:: blosc2.asin
.. autofunction:: blosc2.asinh
.. autofunction:: blosc2.atan
.. autofunction:: blosc2.atan2
.. autofunction:: blosc2.atanh
.. autofunction:: blosc2.bitwise_and
.. autofunction:: blosc2.bitwise_invert
.. autofunction:: blosc2.bitwise_left_shift
.. autofunction:: blosc2.bitwise_or
.. autofunction:: blosc2.bitwise_right_shift
.. autofunction:: blosc2.bitwise_xor
.. autofunction:: blosc2.ceil
.. autofunction:: blosc2.conj
.. autofunction:: blosc2.copysign
.. autofunction:: blosc2.cos
.. autofunction:: blosc2.cosh
.. autofunction:: blosc2.divide
.. autofunction:: blosc2.equal
.. autofunction:: blosc2.exp
.. autofunction:: blosc2.expm1
.. autofunction:: blosc2.floor
.. autofunction:: blosc2.floor_divide
.. autofunction:: blosc2.greater
.. autofunction:: blosc2.greater_equal
.. autofunction:: blosc2.hypot
.. autofunction:: blosc2.isfinite
.. autofunction:: blosc2.isinf
.. autofunction:: blosc2.isnan
.. autofunction:: blosc2.less
.. autofunction:: blosc2.less_equal
.. autofunction:: blosc2.log
.. autofunction:: blosc2.log1p
.. autofunction:: blosc2.log2
.. autofunction:: blosc2.log10
.. autofunction:: blosc2.logaddexp
.. autofunction:: blosc2.logical_and
.. autofunction:: blosc2.logical_not
.. autofunction:: blosc2.logical_or
.. autofunction:: blosc2.logical_xor
.. autofunction:: blosc2.matmul
.. autofunction:: blosc2.maximum
.. autofunction:: blosc2.minimum
.. autofunction:: blosc2.multiply
.. autofunction:: blosc2.negative
.. autofunction:: blosc2.nextafter
.. autofunction:: blosc2.not_equal
.. autofunction:: blosc2.positive
.. autofunction:: blosc2.pow
.. autofunction:: blosc2.reciprocal
.. autofunction:: blosc2.remainder
.. autofunction:: blosc2.sign
.. autofunction:: blosc2.signbit
.. autofunction:: blosc2.sin
.. autofunction:: blosc2.sinh
.. autofunction:: blosc2.sqrt
.. autofunction:: blosc2.square
.. autofunction:: blosc2.subtract
.. autofunction:: blosc2.tan
.. autofunction:: blosc2.tanh
.. autofunction:: blosc2.trunc
.. autofunction:: blosc2.vecdot
