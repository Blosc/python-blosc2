#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterator
from typing import Any

import numpy as np

import blosc2
from blosc2.c2array import C2Array
from blosc2.embed_store import EmbedStore


class ZipStore:
    """
    A directory-based storage container that uses a EmbedStore index for metadata.

    The ZipStore maintains a directory structure with an index file (index.b2t)
    that tracks all stored arrays. It also supports reading from a .b2z file,
    which is a zip archive containing Blosc2 compressed files.

    Parameters
    ----------
    localpath : str
        Local path for the .b2z file. Must have a .b2z extension.
        For reading: The .b2z file must exist.
        For writing: The .b2z file will be created.
    mode : str, optional
        File mode ('r', 'w', 'a'). Default is 'a'.
    tmpdir : str or None, optional
        Temporary directory to use for working files. If None, a system
        temporary directory will be created. Default is None.
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
    >>> zipstore = ZipStore(localpath="my_zipstore.b2z", mode="w")
    >>> zipstore["/node1"] = np.array([1, 2, 3])
    >>> zipstore["/node2"] = blosc2.ones(2)
    >>> print(list(zipstore.keys()))
    ['/node1', '/node2']
    >>> print(zipstore["/node1"][:])
    [1 2 3]
    """

    def __init__(  # noqa: C901
        self,
        localpath: os.PathLike[Any] | str | bytes,
        mode: str = "a",
        tmpdir: str | None = None,
        cparams: blosc2.CParams | None = None,
        dparams: blosc2.CParams | None = None,
        storage: blosc2.Storage | None = None,
    ):
        """
        See :class:`ZipStore` for full documentation of parameters.
        """
        self.offsets = {}
        self.map_tree = {}
        self.localpath = localpath if isinstance(localpath, (str, bytes)) else str(localpath)
        self.mode = mode
        self._temp_dir_obj = None

        # Enforce .b2z extension
        if not self.localpath.endswith(".b2z"):
            raise ValueError("localpath must have a .b2z extension")

        # Handle temporary directory
        if tmpdir is None:
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            self.tmpdir = self._temp_dir_obj.name
        else:
            self.tmpdir = tmpdir
            if not os.path.exists(tmpdir):
                os.makedirs(tmpdir, exist_ok=True)

        if mode == "r":
            # Handle .b2z file input for reading
            if not os.path.exists(self.localpath):
                raise FileNotFoundError(f"Zip file {self.localpath} does not exist.")

            self.b2z_path = self.localpath
            self.index_path = "index.b2t"

            # Populate offsets of files in b2z
            self.offsets = self._get_zip_offsets()

            # Check if index.b2t exists in zip
            if self.index_path not in self.offsets:
                raise FileNotFoundError(f"Index file {self.index_path} not found in b2z file.")

            # Open the index file directly from zip using offset
            index_offset = self.offsets[self.index_path]["offset"]
            schunk = blosc2.open(self.b2z_path, mode="r", offset=index_offset)
            self._estore = EmbedStore(_from_schunk=schunk, _zip_store=True)

            # Build map_tree from .b2nd files in zip
            for filepath in self.offsets:
                if filepath.endswith(".b2nd"):
                    # Convert filename to key: remove .b2nd extension and ensure starts with /
                    key = filepath[:-5]  # Remove .b2nd
                    if not key.startswith("/"):
                        key = "/" + key
                    self.map_tree[key] = filepath
        else:
            # Handle directory input for writing/appending
            self.index_path = os.path.join(self.tmpdir, "index.b2t")

            # Check if we're opening an existing zipstore
            if mode == "a" and os.path.exists(self.localpath):
                # Extract existing .b2z to tmpdir for append mode
                with zipfile.ZipFile(self.localpath, "r") as zf:
                    zf.extractall(self.tmpdir)

            self.b2z_path = self.localpath

            # Initialize the underlying EmbedStore index
            self._estore = EmbedStore(
                urlpath=self.index_path,
                mode=mode,
                cparams=cparams,
                dparams=dparams,
                storage=storage,
                _zip_store=True,
            )

    @property
    def estore(self) -> EmbedStore:
        """
        Access to the underlying EmbedStore index.

        Returns
        -------
        estore : EmbedStore
            The underlying EmbedStore object used for indexing.
        """
        return self._estore

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
        if isinstance(value, blosc2.NDArray) and getattr(value, "urlpath", None):
            # Convert key to a proper file path within the tree directory
            # Remove leading slash and convert to filesystem path
            rel_key = key.lstrip("/")
            # TODO: Handle case where key is root ("/")
            # if not rel_key:  # Handle root key "/"
            #     rel_key = "root"

            # Create the destination path relative to the tree file's directory
            dest_path = os.path.join(self.tmpdir, rel_key + ".b2nd")

            # Ensure the parent directory exists
            parent_dir = os.path.dirname(dest_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            shutil.copy2(value.urlpath, dest_path)
            # Store relative path from tree directory
            rel_path = os.path.relpath(dest_path, self.tmpdir)
            self.map_tree[key] = rel_path
        else:
            self._estore[key] = value

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
        # Check map_tree first (takes precedence over EmbedStore index)
        if key in self.map_tree:
            filepath = self.map_tree[key]
            if filepath in self.offsets:
                offset = self.offsets[filepath]["offset"]
                return blosc2.open(self.b2z_path, mode="r", offset=offset)
            else:
                urlpath = os.path.join(self.tmpdir, filepath)
                if os.path.exists(urlpath):
                    return blosc2.open(urlpath, mode="r" if self.mode == "r" else "a")
                else:
                    raise KeyError(f"File for key '{key}' not found in offsets or temporary directory.")

        # Fall back to EmbedStore index
        return self._estore[key]

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
        return self._estore.get(key, default)

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
        del self._estore[key]

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
        return key in self.map_tree or key in self._estore

    def __len__(self) -> int:
        """
        Return the number of nodes in the zipstore.

        Returns
        -------
        count : int
            Number of nodes.
        """
        return len(self._estore)

    def __iter__(self) -> Iterator[str]:
        """
        Return an iterator over the keys in the zipstore.

        Returns
        -------
        iterator : Iterator[str]
            Iterator over keys.
        """
        return iter(self._estore)

    def keys(self) -> dict[str, dict[str, int]].keys:
        """
        Return all keys in the zipstore.

        Returns
        -------
        keys : dict_keys
            Keys of the zipstore.
        """
        # Combine keys from both map_tree and EmbedStore index
        return set(self.map_tree.keys()) | set(self._estore.keys())

    def values(self) -> Iterator[blosc2.NDArray]:
        """
        Return an iterator over all values in the zipstore.

        Returns
        -------
        values : Iterator[blosc2.NDArray]
            Iterator over stored arrays.
        """
        return self._estore.values()

    def items(self) -> Iterator[tuple[str, blosc2.NDArray]]:
        """
        Return an iterator over (key, value) pairs in the zipstore.

        Returns
        -------
        items : Iterator[tuple[str, blosc2.NDArray]]
            Iterator over key-value pairs.
        """
        # Get all unique keys from both map_tree and _estore, with map_tree taking precedence
        all_keys = set(self.map_tree.keys()) | set(self._estore.keys())

        for key in all_keys:
            # Check map_tree first, then fall back to _estore
            if key in self.map_tree:
                filepath = self.map_tree[key]
                if filepath in self.offsets:
                    offset = self.offsets[filepath]["offset"]
                    yield key, blosc2.open(self.b2z_path, mode="r", offset=offset)
                else:
                    # Fallback if filepath not in offsets
                    yield key, self._estore[key]
            else:
                yield key, self._estore[key]

    def to_b2z(self, overwrite=False) -> os.PathLike[Any] | str:
        """
        Serialize zip store contents to the b2z file.

        Parameters
        ----------
        overwrite : bool, optional
            If True, overwrite the existing b2z file if it exists. Default is False.

        Returns
        -------
        filename : str
            The absolute path to the created b2z file.
        """
        if self.mode == "r":
            raise ValueError("Cannot call to_b2z() on a ZipStore opened in read mode.")

        if os.path.exists(self.b2z_path) and not overwrite:
            raise FileExistsError(f"'{self.b2z_path}' already exists. Use overwrite=True to overwrite.")

        # Gather all files except index_path
        filepaths = []
        for root, _, files in os.walk(self.tmpdir):
            for file in files:
                filepath = os.path.join(root, file)
                if os.path.abspath(filepath) != os.path.abspath(self.index_path):
                    filepaths.append(filepath)

        # Sort filepaths by file size from largest to smallest
        filepaths.sort(key=lambda f: os.path.getsize(f), reverse=True)

        with zipfile.ZipFile(self.b2z_path, "w", zipfile.ZIP_STORED) as zf:
            # Write all files (except index_path) first (sorted by size)
            for filepath in filepaths:
                arcname = os.path.relpath(filepath, self.tmpdir)
                zf.write(filepath, arcname)
            # Write index.b2t last
            if os.path.exists(self.index_path):
                arcname = os.path.relpath(self.index_path, self.tmpdir)
                zf.write(self.index_path, arcname)
        return os.path.abspath(self.b2z_path)

    def _get_zip_offsets(self) -> dict[str, dict[str, int]]:
        """
        Get the offset (and length) of all files in the zip archive.
        """
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

        Yoy always need to call this method to ensure that the zipstore is properly
        created or updated.  Use a context manager to ensure this is called automatically.
        If the zipstore was opened in read mode, this method does nothing.
        """
        if self.mode in ("w", "a"):
            # Serialize to b2z file
            self.to_b2z(overwrite=True)

        # Clean up temporary directory if we created it
        if self._temp_dir_obj is not None:
            self._temp_dir_obj.cleanup()

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
    localpath = "example_zipstore.b2z"
    if True:
        with ZipStore(localpath, mode="w") as zipstore:
            zipstore["/node1"] = np.array([1, 2, 3])
            zipstore["/node2"] = blosc2.ones(2)

            # Make /node3 an external file
            arr_external = blosc2.arange(3, urlpath="ext_node3.b2nd", mode="w")
            zipstore["/dir1/node3"] = arr_external

            print("ZipStore keys:", list(zipstore.keys()))
            print("Node1 data:", zipstore["/node1"][:])
            print("Node2 data:", zipstore["/node2"][:])
            print("Node3 data (external):", zipstore["/dir1/node3"][:])

            del zipstore["/node1"]
            print("After deletion, keys:", list(zipstore.keys()))

    # Open the stored zip file
    with ZipStore(localpath, mode="r") as zipstore_opened:
        print("Opened zipstore keys:", list(zipstore_opened.keys()))
        for key, value in zipstore_opened.items():
            if isinstance(value, blosc2.NDArray):
                print(
                    f"Key: {key}, Shape: {value.shape}, Values: {value[:10] if len(value) > 3 else value[:]}"
                )
