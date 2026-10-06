"""A browsing facade for the independent roots of a Caterva2 service."""

import threading
import weakref
from pathlib import Path

import blosc2
from blosc2.caterva2_url import validate_roots, validate_service_url
from blosc2.info import InfoReporter
from blosc2.proxy_source import Traffic
from blosc2.remote_array import CACHE_POLICY_DEFAULT, RemoteMetadataMapping
from blosc2.remote_store import RemoteNode, RemoteStore


def _close_roots(owners, lock, state):
    with lock:
        state["users"] -= 1
        if state["users"]:
            return
        for store in owners.values():
            store.close()
        owners.clear()


class RemoteRepository(RemoteStore):
    """Read-only, nonpersistent group of Caterva2 roots.

    Returned by ``blosc2.open`` for empty or multi-root services. Root owners
    open lazily. Cache allowances are **per root**, not a repository-wide bound.
    Closing this facade leaves previously returned child handles usable.
    Open a specific root to persist or materialize a selection.
    """

    def __init__(self, urlbase, roots, *, auth_token=None, **options):
        validate_service_url(urlbase)
        self.urlbase = urlbase.rstrip("/")
        self.roots = dict(validate_roots(roots))
        self.auth_token = blosc2.c2array._server_data["auth_token"] if auth_token is None else auth_token
        self.options = dict(options)
        policy, limit = self._validate_cache_config(
            self.options.pop("cache_policy", CACHE_POLICY_DEFAULT),
            self.options.pop("max_cache_bytes", CACHE_POLICY_DEFAULT),
            self.options.get("cache_dir"),
        )
        if self.options.keys() - {"cache_dir"}:
            raise TypeError("RemoteRepository accepts only cache_dir, cache_policy, and max_cache_bytes")
        self.options.update(cache_policy=policy, max_cache_bytes=limit)
        self._lock = threading.RLock()
        self._roots = {}
        self._state = {"users": 1}
        self._traffic = Traffic()
        self._finalizer = weakref.finalize(self, _close_roots, self._roots, self._lock, self._state)
        self.is_tree = True

    def _ensure_open(self):
        if not self._finalizer.alive:
            raise RuntimeError("RemoteRepository is closed")

    def _resolve(self, path):
        raise NotImplementedError("Open a repository root before using this RemoteStore operation")

    def _check_open(self):
        self._ensure_open()

    def _root(self, name):
        self._ensure_open()
        if name not in self.roots:
            raise KeyError(name)
        if name not in self._roots:
            options = self.options.copy()
            if options.get("cache_dir") is not None:
                # The root owner's existing identity hashing includes the base,
                # dataset, and authentication scope. Each root gets its own budget.
                options["cache_dir"] = Path(options["cache_dir"])
            self._roots[name] = RemoteStore(
                blosc2.URLPath(name, urlbase=self.urlbase, auth_token=self.auth_token),
                _traffic=self._traffic,
                **options,
            )
        return self._roots[name]

    def keys(self):
        with self._lock:
            self._ensure_open()
            return sorted(self.roots)

    def __getitem__(self, path):
        with self._lock:
            self._ensure_open()
            if not isinstance(path, str):
                raise TypeError("RemoteRepository paths must be strings")
            relative = path.strip("/")
            if not relative:
                alias = object.__new__(type(self))
                alias.__dict__.update(self.__dict__)
                self._state["users"] += 1
                alias._finalizer = weakref.finalize(
                    alias, _close_roots, self._roots, self._lock, self._state
                )
                return alias
            from blosc2.remote_store import RemoteDiscovery

            RemoteDiscovery._validate(relative)
            name, _, suffix = relative.partition("/")
            return self._root(name)[suffix]

    def get_info(self, path=""):
        with self._lock:
            self._ensure_open()
            if not isinstance(path, str):
                raise TypeError("RemoteRepository paths must be strings")
            relative = path.strip("/")
            from blosc2.remote_store import RemoteDiscovery

            RemoteDiscovery._validate(relative)
            if not relative:
                return RemoteNode("", "group", RemoteMetadataMapping({}))
            name, _, suffix = relative.partition("/")
            if name not in self.roots:
                raise KeyError(name)
            if not suffix:
                # Listing the service must not contact all its roots.
                return RemoteNode(relative, "group", RemoteMetadataMapping(self.roots[name]))
            info = self._root(name).get_info(suffix)
            return RemoteNode(relative, info.kind, info.attrs, info.diagnostic, info.catalog_attrs)

    @property
    def source(self):
        self._ensure_open()
        return {"kind": "caterva2_repository", "version": 1, "urlbase": self.urlbase + "/"}

    @property
    def info(self):
        return InfoReporter(self)

    @property
    def info_items(self):
        return [
            ("type", type(self).__name__),
            ("source", self.source),
            ("roots", self.keys()),
            ("cache allowance", "per root"),
        ]

    @property
    def attrs(self):
        return self.get_info().attrs

    @property
    def cache_bytes(self):
        with self._lock:
            self._ensure_open()
            return sum(store.cache_bytes for store in self._roots.values())

    @property
    def metadata_bytes(self):
        with self._lock:
            self._ensure_open()
            return sum(store.metadata_bytes for store in self._roots.values())

    @property
    def max_cache_bytes(self):
        """Configured allowance per root (not a total repository allowance)."""
        self._ensure_open()
        return self.options["max_cache_bytes"]

    @property
    def cache_policy(self):
        self._ensure_open()
        return self.options["cache_policy"]

    @property
    def traffic(self):
        """Shared counter for opened roots; the initial roots probe is excluded."""
        self._ensure_open()
        return self._traffic

    @property
    def mutable(self):
        self._ensure_open()
        return False

    @mutable.setter
    def mutable(self, value):
        raise NotImplementedError("Repository persistence is unsupported; select a specific root")

    @property
    def is_cache_mutable(self):
        self._ensure_open()
        return True

    def read_cached(self, path, item=(), *, nchunk=None):
        with self._lock:
            self._ensure_open()
            name, _, suffix = path.strip("/").partition("/")
            return self._root(name).read_cached(suffix, item, nchunk=nchunk)

    def read_cached_table(self, operation):
        raise NotImplementedError("Select a specific root for cached table operations")

    def save(self, *args, **kwargs):
        raise NotImplementedError("Repository persistence is unsupported; save a specific root")

    @classmethod
    def with_sparse_cache(cls, *args, **kwargs):
        raise NotImplementedError("Repository shared sparse caching requires selecting a specific root")

    def materialize(self, *args, **kwargs):
        raise NotImplementedError("Repository materialization is unsupported; select a specific root")

    def refresh(self):
        raise NotImplementedError(
            "Reopen the repository to rediscover roots; refresh a specific root separately"
        )

    def close(self):
        self._finalizer()
