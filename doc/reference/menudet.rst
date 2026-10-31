Menudet API
===========

Menudet is the supported language for portable computation on arrays and tables.
For authoring and persistence examples, see :doc:`../guides/menudet`. To replace
the retired alpha decorator, see :doc:`../guides/migrating_to_menudet`.

Menudet 0.1 is **beta**. API and format changes remain possible and will come
with migration guidance.

Key contracts
-------------

* Decorated calls return lazy arrays; ``result[:]`` returns values and
  ``result.compute()`` materializes a Blosc2 array.
* ``MenudetFunction`` knows how to specialize; ``MenudetKernel`` is already
  specialized. Decorated calls specialize automatically; explicit
  ``specialize()`` is for working with the typed kernel itself.
* The Python body is never executed. Native failures propagate without replay
  through Python or the legacy DSL.
* Annotated input dtypes must match exactly. Scalar parameters require dtype
  annotations. Only supported scalar globals and closures can be captured.
* Lazy persistence requires durable operand references and stores validated
  Menudet artifacts, not executable Python function recipes.
* Historical Python-source recipes are unsupported, even with
  ``deserialize="full"``. Materialize them in a previous release before upgrading.

See :doc:`dsl_syntax`, :doc:`menudet_accuracy`, and :doc:`save_load` for syntax,
numerical rules, and deserialization policies. Menudet is not a general sandbox
for arbitrary untrusted source or data.

Specifications
--------------

For the full language and format contracts, see miniexpr's specifications:

* `Menudet 0.1 language specification <https://github.com/Blosc/miniexpr/blob/7ba6d40fdb5084526fc1750d7a30c4125caa332f/doc/dsl-spec/0.1.md>`_
* `Menudet 0.1 artifact specification <https://github.com/Blosc/miniexpr/blob/7ba6d40fdb5084526fc1750d7a30c4125caa332f/doc/dsl-spec/artifact-0.1.md>`_

API
---

.. autofunction:: blosc2.menudet

.. autoclass:: blosc2.MenudetFunction
    :members:

.. autoclass:: blosc2.MenudetKernel
    :members:
    :exclude-members: to_json, from_json

.. autoclass:: blosc2.MenudetError

.. autofunction:: blosc2.validate_menudet
