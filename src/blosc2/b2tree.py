#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from collections.abc import Iterator
from typing import Any

import numpy as np

import blosc2


class Tree:
    """
    A general tree structure for storing numpy/blosc2 arrays as nodes,
    backed by a 1D uint8 blosc2 NDArray. Dictionary-like interface.

    Parameters
    ----------
    urlpath : str or None, optional
        Path for persistent storage.
    mode : str, optional
        File mode ('r', 'w', 'a'). Default is 'w'.
    cparams : dict or None, optional
        Compression parameters.
    dparams : dict or None, optional
        Decompression parameters.
    storage : blosc2.NDArray or None, optional
        Optional existing blosc2 NDArray.

    Examples
    --------
    >>> tree = Tree(urlpath="example_tree.b2z", mode="w")
    >>> tree["/node1"] = np.array([1, 2, 3])
    >>> tree["/node2"] = np.array([[4, 5], [6, 7]])
    >>> tree["/node3"] = np.array([8, 9, 10])
    >>> print(list(tree.keys()))
    ['/node1', '/node2', '/node3']
    """

    def __init__(
        self,
        urlpath: str | None = None,
        mode: str = "w",
        cparams: dict[str, Any] | None = None,
        dparams: dict[str, Any] | None = None,
        storage: blosc2.NDArray | None = None,
    ):
        """
        Initialize the Tree.

        Parameters
        ----------
        urlpath : str or None, optional
            Path for persistent storage.
        mode : str, optional
            File mode ('r', 'w', 'a'). Default is 'w'.
        cparams : dict or None, optional
            Compression parameters.
        dparams : dict or None, optional
            Decompression parameters.
        storage : blosc2.NDArray or None, optional
            Optional existing blosc2 NDArray.
        """
        self._cparams = cparams or {}
        self._dparams = dparams or {}

        if storage is not None:
            self._storage = storage
            self._load_metadata()
        elif mode in ("r", "a") and urlpath:
            self._storage = blosc2.open(urlpath, mode=mode)
            self._load_metadata()
        else:
            initial_data = np.zeros(0, dtype=np.uint8)
            self._storage = blosc2.asarray(
                initial_data, urlpath=urlpath, mode=mode, cparams=self._cparams, dparams=self._dparams
            )
            self._tree_map: dict[str, dict[str, int]] = {}
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
        if required_size > self._storage.shape[0]:
            new_size = max(required_size, int(self._storage.shape[0] * 1.5))
            self._storage.resize((new_size,))

    def __setitem__(self, key: str, value: np.ndarray | blosc2.NDArray) -> None:
        """
        Add a node to the tree.

        Parameters
        ----------
        key : str
            Node key.
        value : np.ndarray or blosc2.NDArray
            Array to store.

        Raises
        ------
        ValueError
            If key is invalid or already exists.
        """
        self._validate_key(key)
        if isinstance(value, np.ndarray):
            value = blosc2.asarray(value, cparams=self._cparams, dparams=self._dparams)
        serialized_data = value.to_cframe()
        data_len = len(serialized_data)
        self._ensure_capacity(data_len)
        offset = self._current_offset
        self._storage[offset : offset + data_len] = np.frombuffer(serialized_data, dtype=np.uint8)
        self._tree_map[key] = {"offset": offset, "length": data_len}
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
        offset = node_info["offset"]
        length = node_info["length"]
        serialized_data = bytes(self._storage[offset : offset + length])
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
        self._storage.vlmeta["tree_metadata"] = metadata

    def _load_metadata(self) -> None:
        """
        Load and deserialize the tree map from the vlmeta.

        Returns
        -------
        None
        """
        if "tree_metadata" in self._storage.vlmeta:
            metadata = self._storage.vlmeta["tree_metadata"]
            self._tree_map = metadata["tree_map"]
            self._current_offset = metadata["current_offset"]
        else:
            self._tree_map = {}
            self._current_offset = 0


if __name__ == "__main__":
    # Example usage
    tree = Tree(urlpath="example_tree.b2z", mode="w")
    tree["/node1"] = np.array([1, 2, 3])
    tree["/node2"] = np.array([[4, 5], [6, 7]])
    tree["/node3"] = np.array([8, 9, 10])

    print("Tree keys:", list(tree.keys()))
    print("Node1 data:", tree["/node1"])
    print("Node2 data:", tree["/node2"])

    del tree["/node1"]
    print("After deletion, keys:", list(tree.keys()))

    # Reading back the tree
    tree_read = Tree(urlpath="example_tree.b2z", mode="r")
    print("Read keys:", list(tree_read.keys()))
    for key, value in tree_read.items():
        print(f"shape of {key}: {value.shape}, dtype: {value.dtype}")
