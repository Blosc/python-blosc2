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
from collections.abc import Iterator, Set
from typing import Any

import numpy as np

import blosc2
from blosc2.c2array import C2Array
from blosc2.embed_store import EmbedStore


class DictStore:
    """
    A directory-based storage container for compressed data using Blosc2.

    Manages a directory-based (`.b2d`) structure of NDArrays and SChunks,
    with an embed store for in-memory data. It also supports creating
    and reading `.b2z` files, which are zip archives that mirror the
    directory structure.

    Parameters
    ----------
    localpath : str
        Local path for the directory (`.b2d`) or file (`.b2z`); other extensions
        are not supported. If a directory is specified, it will be treated as
        a Blosc2 directory format (B2DIR). If a file is specified, it
        will be treated as a Blosc2 zip format (B2ZIP).
    mode : str, optional
        File mode ('r', 'w', 'a'). Default is 'a'.
    tmpdir : str or None, optional
        Temporary directory to use when working with `.b2z` files. If None,
        a system temporary directory will be managed. Default is None.
    cparams : dict or None, optional
        Compression parameters for the internal embed store.
        If None, the default Blosc2 parameters are used.
    dparams : dict or None, optional
        Decompression parameters for the internal embed store.
        If None, the default Blosc2 parameters are used.
    storage : blosc2.Storage or None, optional
        Storage properties for the internal embed store.
        If None, the default Blosc2 storage properties are used.
    threshold : int, optional
        Threshold for the array size (bytes) to be kept in the embed store.
        If the *compressed* array size is below this threshold, it will be
        stored in the embed store instead of as a separate file. If None,
        in-memory arrays are stored in the embed store and on-disk arrays
        are stored as separate files.
        C2Array objects will always be stored in the embed store,
        regardless of their size.
        Default is 2**23 (8 MiB).

    Examples
    --------
    >>> dstore = DictStore(localpath="my_dstore.b2z", mode="w")
    >>> dstore["/node1"] = np.array([1, 2, 3])  # goes to embed store
    >>> dstore["/node2"] = blosc2.ones(2)  # goes to embed store
    >>> arr_external = blosc2.arange(3, urlpath="ext_node3.b2nd", mode="w")
    >>> dstore["/dir1/node3"] = arr_external  # external file in dir1
    >>> dstore.to_b2z()  # persist to the zip file; external files are copied in
    >>> print(list(dstore.keys()))
    ['/node1', '/node2', '/dir1/node3']
    >>> print(dstore["/node1"][:])
    [1 2 3]
    >>> with zipfile.ZipFile("my_dstore.b2z", "r") as zf:
    ...     print(zf.namelist())
    ['dir1/node3.b2nd','embed.b2e']
    """

    def __init__(
        self,
        localpath: os.PathLike[Any] | str | bytes,
        mode: str = "a",
        tmpdir: str | None = None,
        cparams: blosc2.CParams | None = None,
        dparams: blosc2.CParams | None = None,
        storage: blosc2.Storage | None = None,
        threshold: int | None = 2**23,  # Default threshold of 8 MiB (2**23 bytes) for embed store
    ):
        """
        See :class:`DictStore` for full documentation of parameters.
        """
        self.localpath = localpath if isinstance(localpath, (str, bytes)) else str(localpath)
        if not self.localpath.endswith((".b2z", ".b2d")):
            raise ValueError(f"localpath must have a .b2z or .b2d extension; you passed: {self.localpath}")
        if mode not in ("r", "w", "a"):
            raise ValueError("For DictStore containers, mode must be 'r', 'w', or 'a'")

        self.mode = mode
        self.threshold = threshold
        self.cparams = cparams or blosc2.CParams()
        self.dparams = dparams or blosc2.DParams()
        self.storage = storage or blosc2.Storage()

        self.offsets = {}
        self.map_tree = {}
        self._temp_dir_obj = None

        self._setup_paths_and_dirs(tmpdir)

        if self.mode == "r":
            self._init_read_mode()
        else:
            self._init_write_append_mode(cparams, dparams, storage)

    def _setup_paths_and_dirs(self, tmpdir: str | None):
        """Set up working directories and paths."""
        self.is_zip_store = self.localpath.endswith(".b2z")
        if self.is_zip_store:
            if tmpdir is None:
                self._temp_dir_obj = tempfile.TemporaryDirectory()
                self.working_dir = self._temp_dir_obj.name
            else:
                self.working_dir = tmpdir
                os.makedirs(tmpdir, exist_ok=True)
            self.b2z_path = self.localpath
        else:  # .b2d
            self.working_dir = self.localpath
            if self.mode in ("w", "a"):
                os.makedirs(self.working_dir, exist_ok=True)
            self.b2z_path = self.localpath[:-4] + ".b2z"

        self.estore_path = os.path.join(self.working_dir, "embed.b2e")

    def _init_read_mode(self):
        """Initialize the store in read mode."""
        if not os.path.exists(self.localpath):
            raise FileNotFoundError(f"dir/zip file {self.localpath} does not exist.")

        if self.is_zip_store:
            self.offsets = self._get_zip_offsets()
            if "embed.b2e" not in self.offsets:
                raise FileNotFoundError("Embed file embed.b2e not found in store.")
            estore_offset = self.offsets["embed.b2e"]["offset"]
            schunk = blosc2.open(self.b2z_path, mode="r", offset=estore_offset)
            for filepath in self.offsets:
                if filepath.endswith(".b2nd"):
                    key = "/" + filepath[:-5]
                    self.map_tree[key] = filepath
        else:  # .b2d
            if not os.path.isdir(self.localpath):
                raise FileNotFoundError(f"Directory {self.localpath} does not exist for reading.")
            schunk = blosc2.open(self.estore_path, mode="r")
            self._update_map_tree()

        self._estore = EmbedStore(_from_schunk=schunk)

    def _init_write_append_mode(
        self,
        cparams: blosc2.CParams | None,
        dparams: blosc2.DParams | None,
        storage: blosc2.Storage | None,
    ):
        """Initialize the store in write or append mode."""
        if self.mode == "a" and os.path.exists(self.localpath):
            if self.is_zip_store:
                with zipfile.ZipFile(self.localpath, "r") as zf:
                    zf.extractall(self.working_dir)
            elif not os.path.isdir(self.working_dir):
                raise FileNotFoundError(f"Directory {self.working_dir} does not exist for reading.")

        self._estore = EmbedStore(
            urlpath=self.estore_path,
            mode=self.mode,
            cparams=cparams,
            dparams=dparams,
            storage=storage,
        )
        self._update_map_tree()

    def _update_map_tree(self):
        # Build map_tree from .b2nd files in working dir
        for root, _, files in os.walk(self.working_dir):
            for file in files:
                filepath = os.path.join(root, file)
                if filepath.endswith(".b2nd"):
                    # Convert filename to key: remove .b2nd extension and ensure starts with /
                    rel_path = os.path.relpath(filepath, self.working_dir)
                    # Normalize path separators to forward slashes for cross-platform consistency
                    rel_path = rel_path.replace(os.sep, "/")
                    key = rel_path[:-5]  # Remove .b2nd
                    if not key.startswith("/"):
                        key = "/" + key
                    self.map_tree[key] = rel_path

    @property
    def estore(self) -> EmbedStore:
        """
        Access to the underlying EmbedStore object.

        Returns
        -------
        estore : EmbedStore
            The underlying EmbedStore object.
        """
        return self._estore

    def __setitem__(self, key: str, value: np.ndarray | blosc2.NDArray | C2Array) -> None:
        """
        Add a node to the DictStore.

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
        if isinstance(value, np.ndarray):
            value = blosc2.asarray(value, cparams=self.cparams, dparams=self.dparams)
        exceeds_threshold = self.threshold is not None and value.cbytes >= self.threshold
        external_file = isinstance(value, blosc2.NDArray) and getattr(value, "urlpath", None)
        if exceeds_threshold or (external_file and self.threshold is None):
            # Convert key to a proper file path within the tree directory
            rel_key = key.lstrip("/")
            dest_path = os.path.join(self.working_dir, rel_key + ".b2nd")

            # Ensure the parent directory exists
            parent_dir = os.path.dirname(dest_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            # Save the value to the destination path
            if not external_file:
                value.save(urlpath=dest_path)
            else:
                # This should be faster than using value.save() ?
                shutil.copy2(value.urlpath, dest_path)

            # Store relative path from tree directory
            rel_path = os.path.relpath(dest_path, self.working_dir)
            self.map_tree[key] = rel_path
        else:
            if external_file:
                value = blosc2.from_cframe(value.to_cframe())
            self._estore[key] = value

    def __getitem__(self, key: str) -> blosc2.NDArray:
        """
        Retrieve a node from the DictStore.

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
        # Check map_tree first
        if key in self.map_tree:
            filepath = self.map_tree[key]
            if filepath in self.offsets:
                offset = self.offsets[filepath]["offset"]
                return blosc2.open(self.b2z_path, mode="r", offset=offset)
            else:
                urlpath = os.path.join(self.working_dir, filepath)
                if os.path.exists(urlpath):
                    return blosc2.open(urlpath, mode="r" if self.mode == "r" else "a")
                else:
                    raise KeyError(f"File for key '{key}' not found in offsets or temporary directory.")

        # Fall back to EmbedStore
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
        Remove a node from the DictStore.

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
        Check if a key exists in the DictStore.

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
        Return the number of nodes in the DictStore.

        Returns
        -------
        count : int
            Number of nodes.
        """
        return len(self.map_tree) + len(self._estore)

    def __iter__(self) -> Iterator[str]:
        """
        Return an iterator over the keys in the DictStore.

        Returns
        -------
        iterator : Iterator[str]
            Iterator over keys.
        """
        yield from self.map_tree.keys()
        for key in self._estore:
            if key not in self.map_tree:
                yield key
        return iter(self.keys())

    def keys(self) -> Set[str]:
        """
        Return all keys in the DictStore.

        Returns
        -------
        keys : Set[str]
            A set containing all unique keys of the DictStore.
        """
        return self.map_tree.keys() | self._estore.keys()

    def values(self) -> Iterator[blosc2.NDArray]:
        """
        Return an iterator over all values in the DictStore.

        Returns
        -------
        values : Iterator[blosc2.NDArray]
            Iterator over stored arrays.
        """
        return self._estore.values()

    def items(self) -> Iterator[tuple[str, blosc2.NDArray]]:
        """
        Return an iterator over (key, value) pairs in the DictStore.

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
                if self.is_zip_store:
                    if filepath in self.offsets:
                        offset = self.offsets[filepath]["offset"]
                        yield key, blosc2.open(self.b2z_path, mode="r", offset=offset)
                else:
                    urlpath = os.path.join(self.working_dir, filepath)
                    yield key, blosc2.open(urlpath, mode="r" if self.mode == "r" else "a")
            elif key in self._estore:
                yield key, self._estore[key]

    def to_b2z(self, overwrite=False, filename=None) -> os.PathLike[Any] | str:
        """
        Serialize zip store contents to the b2z file.

        Parameters
        ----------
        overwrite : bool, optional
            If True, overwrite the existing b2z file if it exists. Default is False.
        filename : str, optional
            If provided, use this filename instead of the default b2z file path.

        Returns
        -------
        filename : str
            The absolute path to the created b2z file.
        """
        if self.mode == "r":
            raise ValueError("Cannot call to_b2z() on a DictStore opened in read mode.")

        b2z_path = self.b2z_path if filename is None else filename
        if not b2z_path.endswith(".b2z"):
            raise ValueError("b2z_path must have a .b2z extension")

        if os.path.exists(b2z_path) and not overwrite:
            raise FileExistsError(f"'{b2z_path}' already exists. Use overwrite=True to overwrite.")

        # Gather all files except estore_path
        filepaths = []
        for root, _, files in os.walk(self.working_dir):
            for file in files:
                filepath = os.path.join(root, file)
                if os.path.abspath(filepath) != os.path.abspath(self.estore_path):
                    filepaths.append(filepath)

        # Sort filepaths by file size from largest to smallest
        filepaths.sort(key=lambda f: os.path.getsize(f), reverse=True)

        with zipfile.ZipFile(self.b2z_path, "w", zipfile.ZIP_STORED) as zf:
            # Write all files (except estore_path) first (sorted by size)
            for filepath in filepaths:
                arcname = os.path.relpath(filepath, self.working_dir)
                zf.write(filepath, arcname)
            # Write estore last
            if os.path.exists(self.estore_path):
                arcname = os.path.relpath(self.estore_path, self.working_dir)
                zf.write(self.estore_path, arcname)
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

        Yoy always need to call this method to ensure that the DictStore is properly
        created or updated.  Use a context manager to ensure this is called automatically.
        If the DictStore was opened in read mode, this method does nothing.
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
        # No need to handle exceptions, just close the DictStore
        return False


if __name__ == "__main__":
    # Example usage
    localpath = "example_dstore.b2z"
    if True:
        with DictStore(localpath, mode="w") as dstore:
            dstore["/node1"] = np.array([1, 2, 3])
            dstore["/node2"] = blosc2.ones(2)

            # Make /node3 an external file
            arr_external = blosc2.arange(3, urlpath="ext_node3.b2nd", mode="w")
            dstore["/dir1/node3"] = arr_external

            print("DictStore keys:", list(dstore.keys()))
            print("Node1 data:", dstore["/node1"][:])
            print("Node2 data:", dstore["/node2"][:])
            print("Node3 data (external):", dstore["/dir1/node3"][:])

            del dstore["/node1"]
            print("After deletion, keys:", list(dstore.keys()))

    # Open the stored zip file
    with DictStore(localpath, mode="r") as dstore_opened:
        print("Opened dstore keys:", list(dstore_opened.keys()))
        for key, value in dstore_opened.items():
            if isinstance(value, blosc2.NDArray):
                print(
                    f"Key: {key}, Shape: {value.shape}, Values: {value[:10] if len(value) > 3 else value[:]}"
                )
