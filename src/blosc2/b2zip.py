#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import os
import shutil
import zipfile
from collections.abc import Iterator
from typing import Any

import numpy as np

import blosc2
from blosc2.b2tree import Tree
from blosc2.c2array import C2Array


class ZipStore:
    """
    A directory-based storage container that uses a Tree index for metadata.

    The ZipStore maintains a directory structure with an index file (index.b2t)
    that tracks all stored arrays. It also supports reading from a .b2z file,
    which is a zip archive containing Blosc2 compressed files.

    Parameters
    ----------
    dirpath : str
        Directory path for persistent storage or .b2z file for read-only access.
        For directories: The index will be stored as dirpath/index.b2t.
        For .b2z files: Only mode='r' is allowed.
    mode : str, optional
        File mode ('r', 'w', 'a'). Default is 'a'. For .b2z files, only 'r' is allowed.
    cparams : dict or None, optional
        Compression parameters for the index.
        Default is None, which uses the default Blosc2 parameters.
    dparams : dict or None, optional
        Decompression parameters for the index.
        Default is None, which uses the default Blosc2 parameters.
    storage : blosc2.Storage or None, optional
        Storage properties for the index store.

    Examples
    --------
    >>> zipstore = ZipStore(dirpath="my_zipstore", mode="w")
    >>> zipstore["/node1"] = np.array([1, 2, 3])
    >>> zipstore["/node2"] = blosc2.ones(2)
    >>> print(list(zipstore.keys()))
    ['/node1', '/node2']
    >>> print(zipstore["/node1"][:])
    [1 2 3]
    """

    def __init__(  # noqa: C901
        self,
        dirpath: os.PathLike[Any] | str | bytes,
        mode: str = "a",
        cparams: blosc2.CParams | None = None,
        dparams: blosc2.CParams | None = None,
        storage: blosc2.Storage | None = None,
    ):
        """
        See :class:`ZipStore` for full documentation of parameters.
        """
        self.offsets = {}
        self.map_tree = {}
        self.dirpath = dirpath if isinstance(dirpath, (str, bytes)) else str(dirpath)
        self.mode = mode
        self.is_b2z_file = self.dirpath.endswith(".b2z")

        if self.is_b2z_file:
            # Handle .b2z file input
            if mode != "r":
                raise ValueError("Only mode='r' is allowed when opening a .b2z file.")
            if not os.path.exists(self.dirpath):
                raise FileNotFoundError(f"Zip file {self.dirpath} does not exist.")

            self.b2z_path = self.dirpath
            self.index_path = "index.b2t"

            # Populate offsets of files in b2z
            self.offsets = self._get_zip_offsets()

            # Check if index.b2t exists in zip
            if self.index_path not in self.offsets:
                raise FileNotFoundError(f"Index file {self.index_path} not found in b2z file.")

            # Open the index file directly from zip using offset
            index_offset = self.offsets[self.index_path]["offset"]
            schunk = blosc2.open(self.b2z_path, mode="r", offset=index_offset)
            self._tree = Tree(_from_schunk=schunk, _zip_store=True)

            # Build map_tree from .b2nd files in zip
            for filepath in self.offsets:
                if filepath.endswith(".b2nd"):
                    # Convert filename to key: remove .b2nd extension and ensure starts with /
                    key = filepath[:-5]  # Remove .b2nd
                    if not key.startswith("/"):
                        key = "/" + key
                    self.map_tree[key] = filepath
        else:
            # Handle directory input (existing behavior)
            self.index_path = os.path.join(dirpath, "index.b2t")

            # Check if we're opening an existing zipstore
            if mode == "r" and not os.path.exists(self.index_path):
                raise FileNotFoundError(f"ZipStore index file {self.index_path} does not exist for opening.")

            # Handle directory creation/cleanup based on mode
            if mode == "w" and os.path.exists(dirpath):
                # Wipe directory contents for write mode
                shutil.rmtree(dirpath)
                os.makedirs(dirpath, exist_ok=True)
            elif mode in ("w", "a") and not os.path.exists(dirpath):
                # Create directory if it doesn't exist for write/append modes
                os.makedirs(dirpath, exist_ok=True)

            self.b2z_path = f"{self.dirpath}.b2z"

            # Initialize the underlying Tree index
            self._tree = Tree(
                urlpath=self.index_path,
                mode=mode,
                cparams=cparams,
                dparams=dparams,
                storage=storage,
                _zip_store=True,
            )

    @property
    def tree(self) -> Tree:
        """
        Access to the underlying Tree index.

        Returns
        -------
        tree : Tree
            The underlying Tree object used for indexing.
        """
        return self._tree

    def __setitem__(self, key: str, value: np.ndarray | blosc2.NDArray | C2Array) -> None:
        """
        Add a node to the zipstore.

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
        self._tree[key] = value

    def __getitem__(self, key: str) -> blosc2.NDArray:
        """
        Retrieve a node from the zipstore.

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
        # Check map_tree first (takes precedence over Tree index)
        if key in self.map_tree:
            filepath = self.map_tree[key]
            if filepath in self.offsets:
                offset = self.offsets[filepath]["offset"]
                return blosc2.open(self.b2z_path, mode="r", offset=offset)

        # Fall back to Tree index
        return self._tree[key]

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
        return self._tree.get(key, default)

    def __delitem__(self, key: str) -> None:
        """
        Remove a node from the zipstore.

        Parameters
        ----------
        key : str
            Node key.

        Raises
        ------
        KeyError
            If key is not found.
        """
        del self._tree[key]

    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in the zipstore.

        Parameters
        ----------
        key : str
            Node key.

        Returns
        -------
        exists : bool
            True if key exists, False otherwise.
        """
        return key in self.map_tree or key in self._tree

    def __len__(self) -> int:
        """
        Return the number of nodes in the zipstore.

        Returns
        -------
        count : int
            Number of nodes.
        """
        return len(self._tree)

    def __iter__(self) -> Iterator[str]:
        """
        Return an iterator over the keys in the zipstore.

        Returns
        -------
        iterator : Iterator[str]
            Iterator over keys.
        """
        return iter(self._tree)

    def keys(self) -> dict[str, dict[str, int]].keys:
        """
        Return all keys in the zipstore.

        Returns
        -------
        keys : dict_keys
            Keys of the zipstore.
        """
        # Combine keys from both map_tree and Tree index
        return set(self.map_tree.keys()) | set(self._tree.keys())

    def values(self) -> Iterator[blosc2.NDArray]:
        """
        Return an iterator over all values in the zipstore.

        Returns
        -------
        values : Iterator[blosc2.NDArray]
            Iterator over stored arrays.
        """
        return self._tree.values()

    def items(self) -> Iterator[tuple[str, blosc2.NDArray]]:
        """
        Return an iterator over (key, value) pairs in the zipstore.

        Returns
        -------
        items : Iterator[tuple[str, blosc2.NDArray]]
            Iterator over key-value pairs.
        """
        # Get all unique keys from both map_tree and _tree, with map_tree taking precedence
        all_keys = set(self.map_tree.keys()) | set(self._tree.keys())

        for key in all_keys:
            # Check map_tree first, then fall back to _tree
            if key in self.map_tree:
                filepath = self.map_tree[key]
                if filepath in self.offsets:
                    offset = self.offsets[filepath]["offset"]
                    yield key, blosc2.open(self.b2z_path, mode="r", offset=offset)
                else:
                    # Fallback if filepath not in offsets
                    yield key, self._tree[key]
            else:
                # Use the _tree for keys not in map_tree
                yield key, self._tree[key]

    def to_b2z(self, overwrite=False) -> os.PathLike[Any] | str:
        """
        Serialize the zipstore to a b2z file named `dirpath.b2z`.

        Parameters
        ----------
        overwrite : bool, optional
            If True, overwrite the existing b2z file if it exists. Default is False.

        Returns
        -------
        filename : str
            The absolute path to the created b2z file.
        """
        if self.is_b2z_file:
            raise ValueError("Cannot call to_b2z() on a ZipStore opened from a .b2z file.")

        if os.path.exists(self.b2z_path) and not overwrite:
            raise FileExistsError(f"'{self.b2z_path}' already exists. Use overwrite=True to overwrite.")

        # Gather all files except index_path
        filepaths = []
        for root, _, files in os.walk(self.dirpath):
            for file in files:
                filepath = os.path.join(root, file)
                if os.path.abspath(filepath) != os.path.abspath(self.index_path):
                    filepaths.append(filepath)

        # Sort filepaths by file size from largest to smallest
        filepaths.sort(key=lambda f: os.path.getsize(f), reverse=True)

        with zipfile.ZipFile(self.b2z_path, "w", zipfile.ZIP_STORED) as zf:
            # Write all files (except index_path) first (sorted by size)
            for filepath in filepaths:
                arcname = os.path.relpath(filepath, self.dirpath)
                zf.write(filepath, arcname)
            # Write index.b2t last
            if os.path.exists(self.index_path):
                arcname = os.path.relpath(self.index_path, self.dirpath)
                zf.write(self.index_path, arcname)
        return os.path.abspath(self.b2z_path)

    def _get_zip_offsets(self) -> dict[str, dict[str, int]]:
        """
        Get the offset (and length) of all files in the zip archive.
        """
        if not self.is_b2z_file:
            return {}

        self.offsets = {}  # Reset offsets
        with open(self.b2z_path, "rb") as f, zipfile.ZipFile(f) as zf:
            for info in zf.infolist():
                # info.header_offset points to the local file header
                # The actual file data starts after the header
                f.seek(info.header_offset)
                local_header = f.read(30)
                filename_len = int.from_bytes(local_header[26:28], "little")
                extra_len = int.from_bytes(local_header[28:30], "little")
                data_offset = info.header_offset + 30 + filename_len + extra_len
                self.offsets[info.filename] = {"offset": data_offset, "length": info.file_size}
        return self.offsets

    def close(self) -> None:
        """
        Persist changes in the zip file if opened in write or append mode.
        """
        if self.mode in ("w", "a") and not self.is_b2z_file:
            # Serialize to b2z file
            self.to_b2z(overwrite=True)
            # Remove the dirpath directory if it was created
            if os.path.exists(self.dirpath):
                shutil.rmtree(self.dirpath)
        elif self.is_b2z_file:
            # For .b2z files, no need to close anything
            pass

    def __enter__(self):
        """
        Enter the context manager.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.
        """
        self.close()
        # No need to handle exceptions, just close the zipstore
        return False


if __name__ == "__main__":
    # Example usage
    dirpath = "example_zipstore"
    if True:
        zipstore = ZipStore(dirpath=dirpath, mode="w")

        zipstore["/node1"] = np.array([1, 2, 3])
        zipstore["/node2"] = blosc2.ones(2)

        # Make /node3 an external file
        arr_external = blosc2.arange(3, urlpath="external_node3.b2nd", mode="w")
        zipstore["/dir1/node3"] = arr_external

        print("ZipStore keys:", list(zipstore.keys()))
        print("Node1 data:", zipstore["/node1"][:])
        print("Node2 data:", zipstore["/node2"][:])
        print("Node3 data (external):", zipstore["/dir1/node3"][:])

        del zipstore["/node1"]
        print("After deletion, keys:", list(zipstore.keys()))

        zipstore.close()

    # Open the stored zip file
    zip_file = f"{dirpath}.b2z"
    with ZipStore(zip_file, mode="r") as zipstore_opened:
        print("Opened zipstore keys:", list(zipstore_opened.keys()))
        for key, value in zipstore_opened.items():
            print(f"Key: {key}, Shape: {value.shape}, Values: {value[:10] if len(value) > 3 else value[:]}")
