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
Historical Python-specific recipes require explicit ``deserialize="full"``;
the selected policy propagates through nested containers and references.

.. autodata:: DeserializeMode

.. autoexception:: UnsafeDeserializationError
