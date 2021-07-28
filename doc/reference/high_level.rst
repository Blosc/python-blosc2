Super-chunk API
===============

.. currentmodule:: blosc2

SChunk
------

.. autoclass:: SChunk
    :members:
    :undoc-members:
    :exclude-members: __dict__, __weakref__, __dealloc__, __module__, c_schunk
    :special-members:

vlmeta
------
Class to access the variable length metalayers of a super-chunk.
    This class behaves very similarly to a dictionary, and variable length
    metalayers can be appended in the typical way:
       schunk.vlmeta['vlmeta1'] = 'something'
    And can be retrieved similarly:
       value = schunk.vlmeta['vlmeta1']
    Once added attributes cannot be removed.
    This class also honors the `__contains__` and `__len__` special
    functions.  Moreover, a `getall()` method returns all the
    variable length metalayers as a dictionary.

.. autoclass:: vlmeta
    :members:
    :undoc-members:
    :exclude-members:
    :special-members:

Utils
-----

.. autofunction:: remove_urlpath

