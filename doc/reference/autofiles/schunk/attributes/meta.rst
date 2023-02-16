SChunk.meta
===========
Metalayers are small metadata for informing about the properties of data that is stored on a container. NDArray implements its own metalayer on top of C-Blosc2 for storing multidimensional information.

.. currentmodule:: blosc2.SChunk

.. autoclass:: Meta
   :exclude-members: get, keys, items, values

.. currentmodule:: blosc2.SChunk.Meta

Methods
-------

.. autosummary::
    :toctree: autofiles/schunk/attributes/meta
    :nosignatures:

    __getitem__
    __setitem__
    get
    keys
    __iter__
    __contains__
