#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import os.path
from collections.abc import Iterator
from typing import Any

import numpy as np

import blosc2
from blosc2.c2array import C2Array

PROFILE = False  # Set to True to enable PROFILE prints in Tree


class Tree:
    """
    A dictionary-like container for storing NumPy/Blosc2 arrays as nodes.

    For nodes that are stored locally or remotely, only references to the
    arrays are stored, not the arrays themselves. This allows for efficient
    storage and retrieval of large datasets.

    Parameters
    ----------
    urlpath : str or None, optional
        Path for persistent storage.
    mode : str, optional
        File mode ('r', 'w', 'a'). Default is 'w'.
    cparams : dict or None, optional
        Compression parameters for nodes.
        Default is None, which uses the default Blosc2 parameters.
    dparams : dict or None, optional
        Decompression parameters for nodes.
        Default is None, which uses the default Blosc2 parameters.
    storage : blosc2.Storage or None, optional
        Storage properties for the tree store.  If passed, it will override
        the `urlpath` and `mode` parameters.
    initial_size : int, optional
        Initial size of the backing storage in bytes.

    Examples
    --------
    >>> tree = Tree(urlpath="example_tree.b2z", mode="w")
    >>> tree["/node1"] = np.array([1, 2, 3])
    >>> tree["/node2"] = blosc2.ones(2)
    >>> tree["/node3"] = blosc2.arange(3, dtype="i4", urlpath="local_node3.b2nd", mode="w")
    >>> urlpath = blosc2.URLPath("@public/examples/ds-1d.b2nd", "https://cat2.cloud/demo")
    >>> tree["/node4"] = blosc2.open(urlpath, mode="r")
    >>> print(list(tree.keys()))
    ['/node1', '/node2', '/node3', '/node4']
    >>> print(tree["/node1"][:])
    [1 2 3]
    """

    def __init__(
        self,
        urlpath: str | None = None,
        mode: str = "a",
        cparams: blosc2.CParams | None = None,
        dparams: blosc2.CParams | None = None,
        storage: blosc2.Storage | None = None,
        chunksize: int | None = 1024 * 1024,  # Default to 1 MiB
    ):
        """
        See :class:`Tree` for full documentation of parameters.
        """
        self._schunk_store = True  # put this to False to use an NDArray instead of a SChunk
        self.urlpath = urlpath
        self.mode = mode
        self.cparams = cparams or blosc2.CParams()
        # self.cparams.nthreads = 1  # for debugging purposes, use only one thread
        self.dparams = dparams or blosc2.DParams()
        if storage is None:
            self.storage = blosc2.Storage(
                contiguous=True,
                urlpath=urlpath,
                mode=mode,
            )
        else:
            self.storage = storage

        if mode in ("r", "a") and urlpath:
            self._store = blosc2.open(urlpath, mode=mode)
            self._load_metadata()
        else:
            _cparams = self.cparams
            _cparams.typesize = 1  # Ensure typesize is set to 1 for byte storage
            if self._schunk_store:
                self._store = blosc2.SChunk(
                    chunksize=chunksize,
                    data=None,
                    cparams=_cparams,
                    dparams=self.dparams,
                    storage=self.storage,
                )
            else:
                self._store = blosc2.zeros(
                    chunksize,
                    dtype=np.uint8,
                    urlpath=urlpath,
                    mode=mode,
                    cparams=self.cparams,
                    dparams=self.dparams,
                    storage=storage,
                )
            self._tree_map: dict = {}
            self._current_offset = 0

    def _validate_key(self, key: str) -> None:
        """
        Validate the node key.

        Parameters
        ----------
        key : str
            The key to validate.

        Raises
        ------
        TypeError
            If key is not a string.
        ValueError
            If key does not follow the required format or already exists.
        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string.")
        if not key.startswith("/"):
            raise ValueError("Key must start with '/'.")
        if len(key) > 1 and key.endswith("/"):
            raise ValueError("Key cannot end with '/' unless it is the root key '/'.")
        if "//" in key:
            raise ValueError("Key cannot contain consecutive slashes '//'.")
        for char in (":", "\0", "\n", "\r", "\t"):
            if char in key:
                raise ValueError(f"Key cannot contain character: {char!r}")
        if key in self._tree_map:
            raise ValueError(f"Key '{key}' already exists in the tree.")

    def _ensure_capacity(self, needed_bytes: int) -> None:
        """
        Ensure the backing storage has enough capacity.

        Parameters
        ----------
        needed_bytes : int
            Number of bytes needed.
        """
        required_size = self._current_offset + needed_bytes
        if required_size > self._store.shape[0]:
            new_size = max(required_size, int(self._store.shape[0] * 1.5))
            self._store.resize((new_size,))

    def __setitem__(self, key: str, value: np.ndarray | blosc2.NDArray | C2Array) -> None:  # noqa: C901
        """
        Add a node to the tree.

        Parameters
        ----------
        key : str
            Node key.
        value : np.ndarray or blosc2.NDArray or blosc2.C2Array
            Array to store.

        Raises
        ------
        ValueError
            If key is invalid or already exists.
        """
        self._validate_key(key)
        if isinstance(value, blosc2.NDArray) and getattr(value, "urlpath", None):
            self._tree_map[key] = {"urlpath": value.urlpath}
            if PROFILE:
                print(
                    f"3.Current file store size using os.path.getsize: {os.path.getsize(self.urlpath)} bytes"
                )
                print(f"tree_map updated with key '{key}': {self._tree_map}")
        elif isinstance(value, C2Array):
            self._tree_map[key] = {"urlbase": value.urlbase, "path": value.path}
            if PROFILE:
                print(
                    f"4.Current file store size using os.path.getsize: {os.path.getsize(self.urlpath)} bytes"
                )
                print(f"tree_map updated with key '{key}': {self._tree_map}")
        else:
            if isinstance(value, np.ndarray):
                value = blosc2.asarray(value, cparams=self.cparams, dparams=self.dparams)
            serialized_data = value.to_cframe()
            data_len = len(serialized_data)
            if PROFILE:
                print(f"Storing key '{key}' with data length {data_len} bytes.")
                print(
                    f"-1.Current file store size using os.path.getsize: {os.path.getsize(self.urlpath)} bytes"
                )
            if not self._schunk_store:
                self._ensure_capacity(data_len)
            offset = self._current_offset
            if PROFILE:
                print(
                    f"0.Current file store size using os.path.getsize: {os.path.getsize(self.urlpath)} bytes"
                )
            if self._schunk_store:
                self._store[offset : offset + data_len] = serialized_data
            else:
                self._store[offset : offset + data_len] = np.frombuffer(serialized_data, dtype=np.uint8)
            if PROFILE:
                print(
                    f"1.Current file store size using os.path.getsize: {os.path.getsize(self.urlpath)} bytes"
                )
            self._tree_map[key] = {"offset": offset, "length": data_len}
            if PROFILE:
                print(
                    f"2.Current file store size using os.path.getsize: {os.path.getsize(self.urlpath)} bytes"
                )
                print(f"tree_map updated with key '{key}': {self._tree_map}")
            self._current_offset += data_len
        self._save_metadata()

    def __getitem__(self, key: str) -> blosc2.NDArray:
        """
        Retrieve a node from the tree.

        Parameters
        ----------
        key : str
            Node key.

        Returns
        -------
        out : blosc2.NDArray
            The stored array.

        Raises
        ------
        KeyError
            If key is not found.
        """
        if key not in self._tree_map:
            raise KeyError(f"Key '{key}' not found in the tree.")
        node_info = self._tree_map[key]
        urlbase = node_info.get("urlbase", None)
        if urlbase:
            urlpath = blosc2.URLPath(node_info["path"], urlbase=urlbase)
            return blosc2.open(urlpath, mode="r")
        urlpath = node_info.get("urlpath", None)
        if urlpath:
            return blosc2.open(urlpath)
        offset = node_info["offset"]
        length = node_info["length"]
        serialized_data = bytes(self._store[offset : offset + length])
        return blosc2.ndarray_from_cframe(serialized_data)

    def get(self, key: str, default: Any = None) -> blosc2.NDArray | Any:
        """
        Retrieve a node, returning a default value if the key is not found.

        Parameters
        ----------
        key : str
            Node key.
        default : Any, optional
            Value to return if key is not found.

        Returns
        -------
        out : blosc2.NDArray or Any
            The stored array or default value.
        """
        return self[key] if key in self._tree_map else default

    def __delitem__(self, key: str) -> None:
        """
        Remove a node from the tree.

        Parameters
        ----------
        key : str
            Node key.

        Raises
        ------
        KeyError
            If key is not found.
        """
        if key not in self._tree_map:
            raise KeyError(f"Key '{key}' not found in the tree.")
        del self._tree_map[key]
        self._save_metadata()

    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in the tree.

        Parameters
        ----------
        key : str
            Node key.

        Returns
        -------
        exists : bool
            True if key exists, False otherwise.
        """
        return key in self._tree_map

    def __len__(self) -> int:
        """
        Return the number of nodes in the tree.

        Returns
        -------
        count : int
            Number of nodes.
        """
        return len(self._tree_map)

    def __iter__(self) -> Iterator[str]:
        """
        Return an iterator over the keys in the tree.

        Returns
        -------
        iterator : Iterator[str]
            Iterator over keys.
        """
        return iter(self._tree_map)

    def keys(self) -> dict[str, dict[str, int]].keys:
        """
        Return all keys in the tree.

        Returns
        -------
        keys : dict_keys
            Keys of the tree.
        """
        return self._tree_map.keys()

    def values(self) -> Iterator[blosc2.NDArray]:
        """
        Return an iterator over all values in the tree.

        Returns
        -------
        values : Iterator[blosc2.NDArray]
            Iterator over stored arrays.
        """
        for key in self._tree_map:
            yield self[key]

    def items(self) -> Iterator[tuple[str, blosc2.NDArray]]:
        """
        Return an iterator over (key, value) pairs in the tree.

        Returns
        -------
        items : Iterator[tuple[str, blosc2.NDArray]]
            Iterator over key-value pairs.
        """
        for key in self._tree_map:
            yield key, self[key]

    def _save_metadata(self) -> None:
        """
        Serialize and save the tree map to the vlmeta of the storage array.

        Returns
        -------
        None
        """
        metadata = {"tree_map": self._tree_map, "current_offset": self._current_offset}
        self._store.vlmeta["tree_metadata"] = metadata

    def _load_metadata(self) -> None:
        """
        Load and deserialize the tree map from the vlmeta.

        Returns
        -------
        None
        """
        if "tree_metadata" in self._store.vlmeta:
            metadata = self._store.vlmeta["tree_metadata"]
            self._tree_map = metadata["tree_map"]
            self._current_offset = metadata["current_offset"]
        else:
            self._tree_map = {}
            self._current_offset = 0


if __name__ == "__main__":
    # Example usage
    tree = Tree(urlpath="example_tree.b2z", mode="w")  # , cparams=blosc2.CParams(clevel=0))
    tree["/node1"] = np.array([1, 2, 3])
    tree["/node2"] = blosc2.ones(2)
    # Make /node3 an local file
    arr_local = blosc2.arange(3, urlpath="local_node3.b2nd", mode="w")
    tree["/node3"] = arr_local
    urlpath = blosc2.URLPath("@public/examples/ds-1d.b2nd", "https://cat2.cloud/demo")
    arr_remote = blosc2.open(urlpath, mode="r")
    tree["/node4"] = arr_remote

    print("Tree keys:", list(tree.keys()))
    print("Node1 data:", tree["/node1"][:])
    print("Node2 data:", tree["/node2"][:])
    print("Node3 data (local):", tree["/node3"][:])

    del tree["/node1"]
    print("After deletion, keys:", list(tree.keys()))

    # Reading back the tree
    tree_read = Tree(urlpath="example_tree.b2z", mode="r")
    print("Read keys:", list(tree_read.keys()))
    for key, value in tree_read.items():
        print(
            f"shape of {key}: {value.shape}, dtype: {value.dtype}, map: {tree_read._tree_map[key]}, "
            f"values: {value[:10] if len(value) > 3 else value[:]}"
        )
