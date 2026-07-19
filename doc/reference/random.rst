Random
------
Chunk-parallel, NumPy-quality random :ref:`NDArray <NDArray>` constructors. Each chunk
gets its own independent :class:`~numpy.random.SeedSequence`-spawned stream and is
generated concurrently in a thread pool, so generation itself parallelizes (not just
compression).

.. currentmodule:: blosc2.random

.. autosummary::

    default_rng
    Generator

.. autofunction:: blosc2.random.default_rng
.. autoclass:: blosc2.random.Generator
    :members:

NumPy compatibility
~~~~~~~~~~~~~~~~~~~
:class:`Generator` mirrors :class:`numpy.random.Generator` method-for-method: same
method names, same distribution parameters, same math. A few differences apply
consistently across the module:

- ``shape`` replaces numpy's ``size`` and is **required** (no implicit scalar draws),
  and keyword-only on every method except :meth:`~Generator.random`.
- There is no ``out=`` parameter — each call allocates its own destination
  :class:`~blosc2.NDArray`, filled chunk by chunk.
- Extra keyword arguments (``chunks``, ``cparams``, ``urlpath``, ...) are forwarded to
  :func:`blosc2.empty`.
- Results are reproducible for a given ``(seed, call order, shape, chunks)``; changing
  the chunk layout changes the values, since chunks map one-to-one to independent
  ``SeedSequence`` streams.
- :meth:`~Generator.gamma` and :meth:`~Generator.standard_gamma` rename numpy's
  distribution-shape parameter (itself called ``shape``) to ``shape_param``, and make it
  positional-only, to avoid colliding with the array-shape ``shape`` keyword.
- :meth:`~Generator.choice` only supports ``replace=True`` (numpy's default) and a 1-D
  or scalar-int ``a``; sampling without replacement, or along an axis of a
  multi-dimensional ``a``, needs whole-array coordination across chunks that this
  module doesn't do.
- The vector-valued distributions (:meth:`~Generator.dirichlet`,
  :meth:`~Generator.multinomial`, :meth:`~Generator.multivariate_hypergeometric`,
  :meth:`~Generator.multivariate_normal`) draw one length-``k`` vector per element:
  output shape is ``shape + (k,)``, and that trailing dimension is always kept whole
  within a chunk.

Coverage of :class:`numpy.random.Generator`'s public methods:

.. list-table::
   :header-rows: 1
   :widths: 25 10 45

   * - Method
     - Status
     - Notes
   * - ``beta``
     - ✅
     -
   * - ``binomial``
     - ✅
     -
   * - ``bytes``
     - ❌
     - returns raw ``bytes``, not an ``NDArray`` — outside this module's contract
   * - ``chisquare``
     - ✅
     -
   * - ``choice``
     - ✅
     - ``replace=True`` and 1-D/scalar ``a`` only
   * - ``dirichlet``
     - ✅
     - vector output, see above
   * - ``exponential``
     - ✅
     -
   * - ``f``
     - ✅
     -
   * - ``gamma``
     - ✅
     - distribution-shape parameter renamed ``shape_param``
   * - ``geometric``
     - ✅
     -
   * - ``gumbel``
     - ✅
     -
   * - ``hypergeometric``
     - ✅
     -
   * - ``integers``
     - ✅
     -
   * - ``laplace``
     - ✅
     -
   * - ``logistic``
     - ✅
     -
   * - ``lognormal``
     - ✅
     -
   * - ``logseries``
     - ✅
     -
   * - ``multinomial``
     - ✅
     - vector output, see above
   * - ``multivariate_hypergeometric``
     - ✅
     - vector output, see above
   * - ``multivariate_normal``
     - ✅
     - vector output, see above
   * - ``negative_binomial``
     - ✅
     -
   * - ``noncentral_chisquare``
     - ✅
     -
   * - ``noncentral_f``
     - ✅
     -
   * - ``normal``
     - ✅
     -
   * - ``pareto``
     - ✅
     -
   * - ``permutation``
     - ❌
     - whole-array shuffle is inherently sequential, not per-chunk independent
   * - ``permuted``
     - ❌
     - same as ``permutation``
   * - ``poisson``
     - ✅
     -
   * - ``power``
     - ✅
     -
   * - ``random``
     - ✅
     -
   * - ``rayleigh``
     - ✅
     -
   * - ``shuffle``
     - ❌
     - same as ``permutation``
   * - ``standard_cauchy``
     - ✅
     -
   * - ``standard_exponential``
     - ✅
     -
   * - ``standard_gamma``
     - ✅
     - distribution-shape parameter renamed ``shape_param``
   * - ``standard_normal``
     - ✅
     -
   * - ``standard_t``
     - ✅
     -
   * - ``triangular``
     - ✅
     -
   * - ``uniform``
     - ✅
     -
   * - ``vonmises``
     - ✅
     -
   * - ``wald``
     - ✅
     -
   * - ``weibull``
     - ✅
     -
   * - ``zipf``
     - ✅
     -
