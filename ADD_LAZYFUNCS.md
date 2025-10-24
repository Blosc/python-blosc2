# Adding (lazy) functions

Once you have written a (public API) function in Blosc2, it is important to:
* Import it from the relevant module in the ``__init__.py`` file
* Add it to the list of functions in ``__all__`` in the ``__init__.py`` file
* If it is present in numpy, add it to the relevant dictionary (``local_ufunc_map``, ``ufunc_map`` ``ufunc_map_1param``) in ``ndarray.py``

If your function is implemented at the Blosc2 level (and not via either the `LazyUDF` or `LazyExpr` classes), you will need to add some conversion of the inputs to SimpleProxy instances (see e.g. ``matmul`` for an example).

Finally, you also need to deal with it correctly within ``shape_utils.py``.

If the function does not change the shape of the output, simply add it to ``elementwise_funcs`` and you're done.

If the function _does_ change the shape of the output, it is likely either a reduction, a constructor, or a linear algebra function and so should be added to one of those lists (``reducers``, ``constructor`` or ``linalg_funcs``). If the function is a reduction, unless you need to handle an argument that is neither ``axis`` nor ``keepdims``, you don't need to do anything else.
If your function is a constructor, you need to ensure it is handled within the ``visit_Call`` function appropriately (if it has a shape argument this is easy, just add it to the list of functions that has ``zeros, zeros_like`` etc.).

For linear algebra functions it is likely you will have to write a bespoke shape handler within the ``linalg_shape`` function. There is also a list ``linalg_attrs`` for attributes which change the shape (currently only ``T`` and ``mT``) should you need to add one. You will probably need to edit the ``validation_patterns`` list at the top of the ``lazyexpr.py`` file to handle these attributes. Just extend the part that has the negative lookahead "(?!real|imag|T|mT|(".

After this, the imports at the top of the ``lazyexpr.py`` should handle things, where an ``eager_funcs`` list is defined to handle eager execution of functions which change the output shape. Finally, in order to handle name changes between NumPy versions 1 and 2, it may be necessary to add aliases for functions within the blocks defined by ``if NUMPY_GE_2_0:`` in ``lazyexpr.py`` and ``ndarray.py``.
