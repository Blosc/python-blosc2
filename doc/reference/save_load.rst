Save and load
-------------

.. currentmodule:: blosc2

.. autosummary::
    save
    open
    load
    save_array
    load_array
    save_tensor
    load_tensor
    from_cframe

.. autofunction:: save
.. autofunction:: open
.. autofunction:: load
.. autofunction:: save_array
.. autofunction:: load_array
.. autofunction:: save_tensor
.. autofunction:: load_tensor
.. autofunction:: from_cframe

Deserialization policy
~~~~~~~~~~~~~~~~~~~~~~

Validated Menudet artifacts load natively without reconstructing Python code.
Historical source-backed DSL/Python UDF recipes are unsupported under both safe
and full policies. They raise :class:`UnsupportedLegacyRecipeError`; materialize
them using a previous release before upgrading, as described in
:doc:`../guides/migrating_to_menudet`. The selected deserialization policy still
propagates through nested containers and references for supported formats.
Ordinary LazyExpr expressions, including supported numerical calls, reductions,
methods and indexing, load under the default ``deserialize="safe"`` policy
with the existing expression validation. Full-only supported nested operands
(such as proxies) still require explicit opt-in. Legacy table-kernel recipes are
rejected, not silently dropped or reinterpreted. New registrations use
independent-row Menudet semantics.

.. autodata:: DeserializeMode

.. autoexception:: UnsafeDeserializationError

.. autoexception:: UnsupportedLegacyRecipeError
