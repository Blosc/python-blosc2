List of desired features for a 4.0 release
------------------------------------------

* First and foremost, we would like to have at least a basic implementation of the [array API](https://data-apis.org/array-api).  Right now, a lot of low-level work on the basic NDArray container to make indexing work as expected has been done. More work is required in implementing the rest of the API (especially in linear algebra operations).

* Have a completely specified format for the `TreeStore` and `DictStore`.  The format should allow to have containers either in memory or on disk.  Also, it should allow a sparse or contiguous storage.  The user will be able to specify these properties by following the same conventions as for NDArray objects (namely, `urlpath` and `contiguous` params).

    * New `.save()` and `.to_cframe()` methods should be implemented to convert from in-memory representations to on disk and vice-versa.
    * The format for `TreeStore` and `DictStore` will initially be defined at Python level, and documented only in the Python-Blosc2 repository.  An implementation in the C library is desirable, but not mandatory at this point.

* A new `Table` object should be implemented based on the `TreeStore` class (a subclass?), with a label ('table'?) in metalayers indicating that the contents of the tree can be interpreted as a regular table.  As `TreeStore` is hierarchical, a subtree can also be interpreted as a `Table` if there a label in the metalayer of the subtree (or group in HDF5 parlance); that can lead to tables that can have different subtables embedded.  It is not clear yet if we should impose the same number of rows for all the columns.

The constructor for the `Table` object should take some parameters to specify properties:

    * `columnar`: True or False. If True, every column will be stored in a different NDArray object.  If False, the columns will be stored in the same NDArray object, with a compound dtype.  In principle, one should be able to create tables that are hybrid between column and row wise, but at this point it is not clear what is the best way to do that.

`Table` should support at least these methods:

    * `.__getitem__()` and `.__setitem__()` so that values can be get and set.
    * `.append()` for appending (multi-) rows of data for all columns in one go.
    * `.__iter__()` for easy and fast iteration over rows.
    * `.where()`: an iterator for querying with conditions that are evaluated with the internal compute engine.
    * `.index()` for indexing a column and getting better performance in queries (desirable, but optional for 4.0).
