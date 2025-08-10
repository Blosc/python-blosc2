#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import os
from collections.abc import Iterator, MutableMapping

import numpy as np

import blosc2
from blosc2.c2array import C2Array
from blosc2.dict_store import DictStore
from blosc2.ndarray import NDArray


class vlmeta(MutableMapping):
    """
    Class providing access to user metadata on a TreeStore.

    Values are serialized via an in-memory SChunk and persisted into the
    internal EmbedStore of the TreeStore. The actual metadata encoding/decoding
    is delegated to SChunk.vlmeta, mirroring the behavior of blosc2.schunk.vlmeta.
    """

    def __init__(self, tstore: "TreeStore"):
        self._tstore = tstore
        # Use a reserved namespace inside the EmbedStore to avoid colliding with user keys
        self._prefix = "/.vlmeta"

    def _fullkey(self, name: str) -> str:
        if not isinstance(name, (str, bytes, slice)):
            # Let SChunk.vlmeta accept bytes names too (it does internally). For storage key, use str.
            name = str(name)
        if isinstance(name, slice):  # handled by callers
            return self._prefix
        # Ensure valid key for EmbedStore
        return f"{self._prefix}/{name}"

    def _check_writable(self):
        if getattr(self._tstore, "mode", None) == "r":
            raise ValueError("Cannot modify vlmeta in read-only mode.")

    def __setitem__(self, name, content):
        self._check_writable()
        # Support bulk set via [:]
        if isinstance(name, slice):
            if name.start is None and name.stop is None:
                for k, v in content.items():
                    self[k] = v
                return
            raise NotImplementedError("Slicing is not supported, unless [:]")
        # Create a tiny in-memory SChunk and delegate vlmeta set
        schunk = blosc2.SChunk(chunksize=1)
        # Delegate encoding to SChunk.vlmeta which handles msgpack, tuples, etc.
        schunk.vlmeta[name] = content
        # Store the SChunk in the internal EmbedStore under our reserved key
        self._tstore.estore[self._fullkey(name)] = schunk

    def __getitem__(self, name):
        if isinstance(name, slice):
            if name.start is None and name.stop is None:
                return self.getall()
            raise NotImplementedError("Slicing is not supported, unless [:]")
        key = self._fullkey(name)
        # Retrieve the stored SChunk from the EmbedStore
        schunk = self._tstore.estore[key]
        # Delegate decoding to SChunk.vlmeta
        return schunk.vlmeta[name]

    def __delitem__(self, name):
        self._check_writable()
        key = self._fullkey(name)
        del self._tstore.estore[key]

    def __len__(self):
        prefix = self._prefix + "/"
        # Count how many keys in estore belong to the vlmeta namespace
        return sum(1 for k in self._tstore.estore if isinstance(k, str) and k.startswith(prefix))

    def __iter__(self):
        prefix = self._prefix + "/"
        plen = len(prefix)
        for k in self._tstore.estore:
            if isinstance(k, str) and k.startswith(prefix):
                yield k[plen:]

    def getall(self):
        """
        Return all the variable length metalayers as a dictionary
        """
        return {name: self[name] for name in self}

    def __repr__(self):
        return repr(self.getall())

    def __str__(self):
        return str(self.getall())


class TreeStore(DictStore):
    """
    A hierarchical tree-based storage container for compressed data using Blosc2.

    Extends DictStore with strict hierarchical key validation and tree traversal
    capabilities. Keys must follow a hierarchical structure using '/' as separator
    and always start with '/'.

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
    >>> tstore = TreeStore(localpath="my_tstore.b2z", mode="w")
    >>> # Create a hierarchy. Data is stored in leaf nodes.
    >>> # Structural nodes like /child0 and /child0/child1 are created automatically.
    >>> tstore["/child0/leaf1"] = np.array([1, 2, 3])
    >>> tstore["/child0/child1/leaf2"] = np.array([4, 5, 6])
    >>> tstore["/child0/child2"] = np.array([7, 8, 9])
    >>>
    >>> # Walk the tree structure
    >>> for path, children, nodes in tstore.walk("/child0"):
    ...     print(f"Path: {path}, Children: {sorted(children)}, Nodes: {sorted(nodes)}")
    Path: /child0, Children: ['/child0/child1'], Nodes: ['/child0/child2', '/child0/leaf1']
    Path: /child0/child1, Children: [], Nodes: ['/child0/child1/leaf2']
    >>>
    >>> # Get a subtree view
    >>> subtree = tstore.get_subtree("/child0")
    >>> sorted(list(subtree.keys()))
    ['/child1/leaf2', '/child2', '/leaf1']
    """

    def __init__(self, *args, _from_parent_store=None, **kwargs):
        """Initialize TreeStore with subtree support."""
        if _from_parent_store is not None:
            # This is a subtree view, copy state from parent
            self.__dict__.update(_from_parent_store.__dict__)
        else:
            super().__init__(*args, **kwargs)
            self.subtree_path = ""  # Empty string means full tree

    def _translate_key_to_full(self, key: str) -> str:
        """Translate a subtree-relative key to a full tree key."""
        if not self.subtree_path:
            return key
        if key == "/":
            return self.subtree_path
        else:
            return self.subtree_path + key

    def _translate_key_from_full(self, full_key: str) -> str | None:
        """Translate a full tree key to a subtree-relative key."""
        if not self.subtree_path:
            return full_key
        if full_key == self.subtree_path:
            return "/"
        elif full_key.startswith(self.subtree_path + "/"):
            return full_key[len(self.subtree_path) :]
        else:
            # Key is not within this subtree
            return None

    def _validate_key(self, key: str) -> None:
        """
        Validate that a key follows hierarchical structure rules.

        Parameters
        ----------
        key : str
            The key to validate.

        Raises
        ------
        ValueError
            If key doesn't follow hierarchical rules.
        """
        if not isinstance(key, str):
            raise ValueError(f"Key must be a string, got {type(key)}")

        if not key.startswith("/"):
            raise ValueError(f"Key must start with '/', got: {key}")

        if key != "/" and key.endswith("/"):
            raise ValueError(f"Key cannot end with '/' (except for root), got: {key}")

        if "//" in key:
            raise ValueError(f"Key cannot contain empty path segments '//', got: {key}")

        # Additional validation for special characters that might cause issues
        invalid_chars = ["\0", "\n", "\r", "\t"]
        for char in invalid_chars:
            if char in key:
                raise ValueError(f"Key cannot contain invalid character {char!r}, got: {key}")

    def __setitem__(self, key: str, value: np.ndarray | NDArray | C2Array) -> None:
        """
        Add a node to the TreeStore with hierarchical key validation.

        Parameters
        ----------
        key : str
            Hierarchical node key (must start with '/' and use '/' as separator).
        value : np.ndarray or blosc2.NDArray or blosc2.C2Array
            Array to store.

        Raises
        ------
        ValueError
            If key doesn't follow hierarchical structure rules, if trying to
            assign to a structural path that already has children, or if trying
            to add a child to a path that already contains data.
        """
        self._validate_key(key)

        # Check if this key already has children (is a structural subtree)
        children = self.get_children(key)
        if children:
            raise ValueError(
                f"Cannot assign array to structural path '{key}' that already has children: {children}"
            )

        # Check if we're trying to add a child to a path that already has data
        # Extract parent path from the key
        if key != "/":
            parent_path = "/".join(key.split("/")[:-1])
            if not parent_path:  # Handle case where parent is root
                parent_path = "/"

            full_parent_key = self._translate_key_to_full(parent_path)
            if super().__contains__(full_parent_key):
                raise ValueError(
                    f"Cannot add child '{key}' to path '{parent_path}' that already contains data"
                )

        full_key = self._translate_key_to_full(key)
        super().__setitem__(full_key, value)

    def __getitem__(self, key: str) -> "NDArray | TreeStore":
        """
        Retrieve a node from the TreeStore with key validation.

        If the key points to a subtree (intermediate path with children),
        returns a TreeStore view of that subtree. If the key points to
        a final node (leaf), returns the stored array.

        Parameters
        ----------
        key : str
            Hierarchical node key.

        Returns
        -------
        out : blosc2.NDArray or TreeStore
            The stored array if key is a leaf node, or a TreeStore subtree view
            if key is an intermediate path with children.

        Raises
        ------
        KeyError
            If key is not found.
        ValueError
            If key doesn't follow hierarchical structure rules.
        """
        self._validate_key(key)
        full_key = self._translate_key_to_full(key)

        # Check if this key has children (is a subtree)
        children = self.get_children(key)

        # Check if the key exists as an actual data node
        key_exists_as_data = super().__contains__(full_key)

        if children:
            # If it has children, return a subtree view
            return self.get_subtree(key)
        elif key_exists_as_data:
            # If no children but exists as data, it's a leaf node - get the actual data
            return super().__getitem__(full_key)
        else:
            # Key doesn't exist at all
            raise KeyError(f"Key '{key}' not found")

    def __delitem__(self, key: str) -> None:
        """
        Remove a node or subtree from the TreeStore with key validation.

        If the key points to a subtree (intermediate path with children),
        removes all nodes in that subtree recursively. If the key points to a final
        node (leaf), removes only that node.

        Parameters
        ----------
        key : str
            Hierarchical node key.

        Raises
        ------
        KeyError
            If key is not found.
        ValueError
            If key doesn't follow hierarchical structure rules.
        """
        self._validate_key(key)

        # Check if the key exists (either as data or as a structural node with descendants)
        full_key = self._translate_key_to_full(key)
        key_exists_as_data = super().__contains__(full_key)
        descendants = self.get_descendants(key)

        if not key_exists_as_data and not descendants:
            raise KeyError(f"Key '{key}' not found")

        # Collect all keys to delete (leaf nodes only, since structural nodes don't exist as data)
        keys_to_delete = []

        # If the key itself has data, include it
        if key_exists_as_data:
            keys_to_delete.append(key)

        # Add all descendant leaf nodes (only those that actually exist as data)
        for descendant in descendants:
            full_descendant_key = self._translate_key_to_full(descendant)
            if super().__contains__(full_descendant_key):
                keys_to_delete.append(descendant)

        # Delete all data keys in the subtree
        for k in keys_to_delete:
            full_key_to_delete = self._translate_key_to_full(k)
            super().__delitem__(full_key_to_delete)

    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in the TreeStore with validation.

        Parameters
        ----------
        key : str
            Hierarchical node key.

        Returns
        -------
        exists : bool
            True if key exists, False otherwise.
        """
        try:
            self._validate_key(key)
            full_key = self._translate_key_to_full(key)
            return super().__contains__(full_key)
        except ValueError:
            return False

    def keys(self):
        """Return all keys in the current subtree view."""
        if not self.subtree_path:
            all_keys = set(super().keys())
        else:
            all_keys = set()
            for full_key in super().keys():  # noqa: SIM118
                relative_key = self._translate_key_from_full(full_key)
                if relative_key is not None:
                    all_keys.add(relative_key)

        # Also include structural paths (intermediate nodes that have children but no data)
        structural_keys = set()
        for key in all_keys:
            # For each leaf key, add all its parent paths
            parts = key.split("/")[1:]  # Remove empty first element from split
            current_path = ""
            for part in parts[:-1]:  # Exclude the leaf itself
                current_path = current_path + "/" + part if current_path else "/" + part
                if current_path and current_path != "/" and current_path not in all_keys:
                    structural_keys.add(current_path)

        return all_keys | structural_keys

    def items(self) -> Iterator[tuple[str, "NDArray | TreeStore"]]:
        """Return key-value pairs in the current subtree view."""
        for key in self.keys():
            yield key, self[key]

    def get_children(self, path: str) -> list[str]:
        """
        Get direct children of a given path within the current subtree.

        Parameters
        ----------
        path : str
            The parent path to get children for.

        Returns
        -------
        children : list[str]
            List of direct child paths.
        """
        self._validate_key(path)

        if path == "/":
            prefix = "/"
        else:
            prefix = path + "/"

        prefix_len = len(prefix)
        children_names = set()

        for key in self.keys():
            if key.startswith(prefix):
                # e.g. key = /hierarchy/level1/data, prefix = /hierarchy/
                # rest = level1/data
                rest = key[prefix_len:]
                # child_name = level1
                child_name = rest.split("/")[0]
                children_names.add(child_name)

        if path == "/":
            return sorted(["/" + name for name in children_names])
        else:
            return sorted([path + "/" + name for name in children_names])

    def get_descendants(self, path: str) -> list[str]:
        """
        Get all descendants of a given path within the current subtree.

        Parameters
        ----------
        path : str
            The parent path to get descendants for.

        Returns
        -------
        descendants : list[str]
            List of all descendant paths.
        """
        self._validate_key(path)

        if path == "/":
            prefix = "/"
        else:
            prefix = path + "/"

        descendants = set()

        # Get all leaf nodes under this path
        for key in self.keys():
            if key.startswith(prefix) and key != path:
                descendants.add(key)

        # Also get all intermediate structural paths
        for key in self.keys():
            if key.startswith(prefix) and key != path:
                # For each leaf node, add all its parent paths
                parts = key[len(prefix) :].split("/")
                current_path = path
                for part in parts[:-1]:  # Exclude the leaf itself
                    current_path = current_path + "/" + part if current_path != "/" else "/" + part
                    if current_path != path:
                        descendants.add(current_path)

        return sorted(descendants)

    def walk(self, path: str = "/") -> Iterator[tuple[str, list[str], list[str]]]:
        """
        Walk the tree structure top-down within the current subtree.

        Similar to os.walk(), this visits all structural nodes in the hierarchy,
        yielding information about each level. Returns relative names, not full paths.

        Parameters
        ----------
        path : str, optional
            The root path to start walking from. Default is "/".

        Yields
        ------
        path : str
            Current path being walked.
        children : list[str]
            List of child directory names (structural nodes that have descendants).
            These are just the names, not full paths.
        nodes : list[str]
            List of leaf node names (nodes that contain data).
            These are just the names, not full paths.

        Examples
        --------
        >>> for path, children, nodes in tstore.walk("/child0"):
        ...     print(f"Path: {path}, Children: {children}, Nodes: {nodes}")
        """
        self._validate_key(path)

        # Get all direct children of this path
        direct_children = self.get_children(path)

        # Separate children into directories (have descendants) and leaf nodes
        children_dirs = []
        leaf_nodes = []

        for child in direct_children:
            child_descendants = self.get_descendants(child)
            if child_descendants:
                # Extract just the name from the full path
                child_name = child.split("/")[-1]
                children_dirs.append(child_name)
            else:
                # Extract just the name from the full path
                child_name = child.split("/")[-1]
                leaf_nodes.append(child_name)

        # Yield current level (always yield the current path being walked)
        yield path, children_dirs, leaf_nodes

        # Recursively walk child directories (structural nodes)
        for child in direct_children:
            child_descendants = self.get_descendants(child)
            if child_descendants:
                yield from self.walk(child)

    def get_subtree(self, path: str) -> "TreeStore":
        """
        Create a view of a subtree with the specified path as the new root.

        Parameters
        ----------
        path : str
            The path that will become the root of the subtree view (relative to current subtree).

        Returns
        -------
        subtree : TreeStore
            A new TreeStore instance that presents the subtree as if `path` were the root.

        Examples
        --------
        >>> tstore["/child0/child1/data"] = np.array([1, 2, 3])
        >>> tstore["/child0/child1/grandchild"] = np.array([4, 5, 6])
        >>> subtree = tstore.get_subtree("/child0/child1")
        >>> list(subtree.keys())
        ['/data', '/grandchild']
        >>> subtree["/grandchild"][:]
        array([4, 5, 6])
        """
        self._validate_key(path)
        full_path = self._translate_key_to_full(path)

        # Create a new TreeStore instance that shares the same underlying storage
        # but with a different subtree_path
        subtree = TreeStore(_from_parent_store=self)
        subtree.subtree_path = full_path

        return subtree

    @property
    def vlmeta(self) -> vlmeta:
        """
        Access to variable-length metadata for the TreeStore.

        Notes
        -----
        Each vlmeta entry is stored as a tiny in-memory SChunk inside the
        internal EmbedStore, and the actual metadata is delegated to the
        SChunk.vlmeta mechanism. This mirrors SChunk.vlmeta behavior and
        ensures robust serialization.
        """
        return vlmeta(self)


if __name__ == "__main__":
    # Example usage
    localpath = "example_tstore.b2z"

    with TreeStore(localpath, mode="w") as tstore:
        # Create a hierarchical structure.
        # Note: data is stored in leaf nodes, not structural nodes.
        tstore["/child0/data_node"] = np.array([1, 2, 3])
        tstore["/child0/child1/data_node"] = np.array([4, 5, 6])
        tstore["/child0/child2"] = np.array([7, 8, 9])
        tstore["/child0/child1/grandchild"] = np.array([10, 11, 12])
        tstore["/other"] = np.array([13, 14, 15])

        print("TreeStore keys:", sorted(tstore.keys()))

        # Test subtree view
        root_subtree = tstore["/child0"]
        root_subtree.vlmeta["foo"] = "bar"
        print("Subtree keys:", sorted(root_subtree.keys()))
        print("Subtree vlmeta:", root_subtree.vlmeta)

        # Walk the tree
        for path, children, nodes in root_subtree.walk("/"):
            print(f"Path: {path}, Children: {children}, Nodes: {nodes}")

    # Clean up
    if os.path.exists(localpath):
        os.remove(localpath)
