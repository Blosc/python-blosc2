"""Service URL recognition and repository discovery (offline loopback fixtures)."""

import pytest
from test_remote_caterva2 import caterva2_source  # noqa: F401

import blosc2
from blosc2.caterva2_url import caterva2_urlpath


@pytest.mark.parametrize(
    ("url", "base", "path"),
    [
        ("http://localhost:8000/@public", "http://localhost:8000", "@public"),
        ("http://[::1]:8000/demo/@public/a/", "http://[::1]:8000/demo", "@public/a"),
        (
            "https://host/deploy%20here/%40public/my%20array/@nested",
            "https://host/deploy%20here",
            "@public/my array/@nested",
        ),
    ],
)
def test_root_marker(url, base, path):
    target = caterva2_urlpath(url)
    assert target.urlbase == base
    assert target.path == path


@pytest.mark.parametrize(
    "url", ["https://host/file.h5", "https://host/data?x=@public", "s3://bucket/@public/a"]
)
def test_ordinary_url(url):
    assert caterva2_urlpath(url) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://host/@",
        "https://host/@public//a",
        "https://host/../@public",
        "https://host/@public/%2F",
        "https://host/@public/%252F",
        "https://host/@public/a%GG",
        "https://host/@public/%FF",
        "https://host/@public/a?x=1",
        "https://host/@public/a#x",
        "https://user:pass@host/@public",
        "https://host/@public/a::/member",
        "https://host/@public/%00",
    ],
)
def test_unsafe_shorthand(url):
    with pytest.raises((ValueError, UnicodeError)):
        caterva2_urlpath(url)


def test_shorthand_open(caterva2_source):  # noqa: F811
    base, array, _, _ = caterva2_source
    with blosc2.open(base + "@public/group") as group:
        assert isinstance(group, blosc2.RemoteStore)
        assert group.keys() == ["array", "table"]
        with group["array"] as remote:
            assert (remote[1:2, 2:4] == array[1:2, 2:4]).all()
    with blosc2.open(base + "@public/table") as table:
        assert isinstance(table, blosc2.RemoteCTable)
        assert table.slice(0, 2).nrows == 2
    assert isinstance(blosc2.open(base + "@public/group/array", lazy=False), blosc2.C2Array)


def test_override_and_option_validation(monkeypatch):
    monkeypatch.setattr(blosc2.schunk, "_open_fsspec_url", lambda *args: "ordinary-file")
    assert blosc2.open("https://host/@public/data.b2nd", remote_service="fsspec") == "ordinary-file"
    for options in ({"remote_service": "wrong"}, {"dataset": "a"}, {"hdf5_index": {}}):
        with pytest.raises(ValueError):
            blosc2.open("https://host/@public/a", **options)
    with pytest.raises(ValueError):
        blosc2.open(blosc2.URLPath("@public", urlbase="https://host"), remote_service="fsspec")


def test_mount_expansion_is_lazy_and_memoized(caterva2_source):  # noqa: F811
    base, array, _, stats = caterva2_source
    with blosc2.open(base + "@public") as store:
        assert stats["requests"] == ["/api/info/@public"]
        assert dict(store.get_info().catalog_attrs) == {"note": "curated"}
        assert dict(store.attrs) == {"name": "fixture"}
        assert store.keys() == ["mount"]
        assert "/api/list/@public/mount" not in stats["requests"]
        with store["mount"] as mount:
            assert mount.keys() == ["array", "empty", "table"]
            assert mount.kind("array") == "ndarray"
            with mount["empty"] as empty:
                assert empty.keys() == []
            before = stats["requests"].copy()
            assert mount.keys() == ["array", "empty", "table"]
            assert stats["requests"] == before
        with store["mount/array"] as remote:
            assert (remote[0:1, 0:2] == array[0:1, 0:2]).all()


def test_direct_lookup_and_recursive_list(caterva2_source):  # noqa: F811
    base, _, _, stats = caterva2_source
    stats["groups"]["@public"] = ["mount/array", "mount/table", "mount/empty", "mount/array"]
    with blosc2.open(base + "@public") as store:
        assert store.kind("mount/array") == "ndarray"
        assert not any("/api/list/" in request for request in stats["requests"])
        assert store.keys() == ["mount"]
        assert "/api/info/@public/mount/table" not in stats["requests"]
        with store["mount"] as mount:
            assert mount.keys() == ["array", "empty", "table"]


def test_listing_failure_can_retry_and_limits_are_atomic(caterva2_source):  # noqa: F811
    import httpx

    base, _, _, stats = caterva2_source
    with blosc2.RemoteStore(blosc2.URLPath("@public", urlbase=base), _max_nodes=5) as store:
        assert store.keys() == ["mount"]
        stats["fail_list"] = "@public/mount"
        with store["mount"] as mount:
            with pytest.raises(httpx.HTTPStatusError):
                mount.keys()
            stats["fail_list"] = None
            previous = store._owner.nodes.copy()
            with pytest.raises(ValueError, match="node limit"):
                mount.keys()
            assert store._owner.nodes == previous
            stats["groups"]["@public/mount"] = ["empty"]
            assert mount.keys() == ["empty"]


def test_concurrent_expansion_is_coalesced(caterva2_source):  # noqa: F811
    from concurrent.futures import ThreadPoolExecutor

    base, _, _, stats = caterva2_source
    with blosc2.open(base + "@public") as store, ThreadPoolExecutor(4) as executor:
        results = list(executor.map(lambda _: store.keys(), range(12)))
        assert results == [["mount"]] * 12
        assert stats["requests"].count("/api/list/@public") == 1


def test_bare_and_prefixed_service(caterva2_source):  # noqa: F811
    base, _, _, stats = caterva2_source
    for url in (base.rstrip("/"), base + "demo"):
        with blosc2.open(url) as store:
            assert isinstance(store, blosc2.RemoteStore)
            assert not isinstance(store, blosc2.RemoteRepository)
            assert store.keys() == ["mount"]
    assert stats["requests"].count("/api/roots") == 2


def test_repository_is_lazy_and_children_outlive_it(caterva2_source, tmp_path):  # noqa: F811
    base, array, _, stats = caterva2_source
    stats["roots"]["@broken"] = {"name": "@broken"}
    repo = blosc2.open(base, cache_dir=tmp_path / "cache")
    try:
        assert isinstance(repo, blosc2.RemoteRepository)
        assert repo.keys() == ["@broken", "@public"]
        assert stats["requests"] == ["/api/roots"]
        assert repo.kind("@broken") == "group"
        with pytest.raises(KeyError):
            repo["@broken"]
        with repo[""] as alias:
            child = alias["@public/mount/array"]
            assert child is not None
        with pytest.raises(NotImplementedError, match="persistence"):
            repo.save(tmp_path / "repository.b2z")
        assert repo.cache_policy is blosc2.CachePolicy.DISK
        assert repo.cache_bytes == 0
        assert "per root" in str(repo.info)
    finally:
        repo.close()
    with child:
        assert (child[0:1, 0:2] == array[0:1, 0:2]).all()
    with pytest.raises(RuntimeError, match="closed"):
        repo.keys()


def test_empty_service_and_repository_option_validation(caterva2_source):  # noqa: F811
    base, _, _, stats = caterva2_source
    stats["roots"] = {}
    with blosc2.open(base) as repo:
        assert repo.keys() == []
        assert repo.kind() == "group"
    for options in (
        {"lazy": False},
        {"cache_path": "a.b2nd"},
        {"shared_cache": True},
        {"storage_options": {}, "remote_service": "caterva2"},
    ):
        with pytest.raises((ValueError, NotImplementedError)):
            blosc2.open(base, **options)


def test_direct_format_does_not_probe(monkeypatch):
    monkeypatch.setattr(blosc2.schunk, "_open_fsspec_url", lambda *args: "ordinary-file")
    monkeypatch.setattr(
        blosc2.caterva2_url, "discover_service", lambda *args, **kwargs: pytest.fail("unexpected probe")
    )
    for url in (
        "https://host/a.b2nd",
        "https://host/a.b2",
        "https://host/a.zarr/",
        "https://host/a.h5::/group",
        "https://host/a.b2z?version=2",
    ):
        assert blosc2.open(url) == "ordinary-file"


@pytest.mark.parametrize("status", [401, 403, 500, 503])
def test_discovery_preserves_http_failure(caterva2_source, status):  # noqa: F811
    import httpx

    base, _, _, stats = caterva2_source
    stats["roots_status"] = status
    with pytest.raises(httpx.HTTPStatusError) as error:
        blosc2.open(base)
    assert error.value.response.status_code == status
    assert stats["requests"] == ["/api/roots"]


@pytest.mark.parametrize("roots", [[], {"../evil": {}}, {"@public": "bad"}, {"@public": {"name": "other"}}])
def test_invalid_discovery_falls_back_only_in_auto(caterva2_source, monkeypatch, roots):  # noqa: F811
    base, _, _, stats = caterva2_source
    stats["roots"] = roots
    monkeypatch.setattr(blosc2.schunk, "_open_fsspec_url", lambda *args: "ordinary-file")
    assert blosc2.open(base) == "ordinary-file"
    with pytest.raises(ValueError, match="roots response"):
        blosc2.open(base, remote_service="caterva2")


def test_discovery_redirects_timeout_and_response_limits(monkeypatch):
    import httpx

    from blosc2.caterva2_url import discover_service

    cases = [
        (
            lambda request: httpx.Response(302, headers={"location": "https://elsewhere/api/roots"}),
            ValueError,
        ),
        (lambda request: httpx.Response(200, content=b"x" * ((1 << 20) + 1)), ValueError),
        (lambda request: httpx.Response(302, headers={"location": str(request.url)}), ValueError),
    ]
    for handler, error in cases:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            monkeypatch.setattr(blosc2.c2array, "_sync_client", lambda: client)
            with pytest.raises(error):
                discover_service("https://host/demo", auth_token="secret")

    def timeout(request):
        raise httpx.ReadTimeout("slow service", request=request)

    with httpx.Client(transport=httpx.MockTransport(timeout)) as client:
        monkeypatch.setattr(blosc2.c2array, "_sync_client", lambda: client)
        with pytest.raises(httpx.ReadTimeout):
            discover_service("https://host")

    seen = []

    def redirect(request):
        seen.append(str(request.url))
        return (
            httpx.Response(302, headers={"location": "/demo/api/roots/"})
            if len(seen) == 1
            else httpx.Response(200, json={"@public": {"name": "@public"}})
        )

    with httpx.Client(transport=httpx.MockTransport(redirect)) as client:
        monkeypatch.setattr(blosc2.c2array, "_sync_client", lambda: client)
        assert discover_service("https://host/demo") == {"@public": {"name": "@public"}}
    assert seen == ["https://host/demo/api/roots", "https://host/demo/api/roots/"]


def test_repository_freezes_auth_context(caterva2_source):  # noqa: F811
    base, _, _, stats = caterva2_source
    stats["roots"]["@broken"] = {"name": "@broken"}
    with blosc2.c2context(auth_token="alice=secret"):
        repo = blosc2.open(base)
    with blosc2.c2context(auth_token="bob=secret"), repo:
        with repo["@public"] as root:
            assert root.keys() == ["mount"]
    assert set(stats["cookies"]) == {"alice=secret"}


def test_nonlazy_groups_and_lookup_escaping_fail_clearly(caterva2_source):  # noqa: F811
    base, _, _, stats = caterva2_source
    with pytest.raises(NotImplementedError, match="requires lazy=True"):
        blosc2.open(base + "@public", lazy=False)
    with blosc2.open(base + "@public") as store:
        before = stats["requests"].copy()
        for path in ("mount/array?x=1", "mount/array#x", "mount/array%2F"):
            with pytest.raises(ValueError, match="Unsafe"):
                store.get_info(path)
        assert stats["requests"] == before
        with pytest.raises(KeyError):
            store["nonexistent"]


def test_old_catalog_listing_cache_is_rediscovered(caterva2_source, tmp_path):  # noqa: F811
    base, _, _, _ = caterva2_source
    cache = tmp_path / "cache"
    with blosc2.open(base + "@public", cache_dir=cache) as store:
        with store["mount"] as mount:
            assert mount.keys() == ["array", "empty", "table"]
        manifest = store._owner.disk.load()
        manifest["metadata"].pop("caterva2_listing_version")
        manifest["listed"]["@public/mount"] = []
        store._owner.disk.publish(manifest)
    with blosc2.open(base + "@public", cache_dir=cache) as reopened:
        with reopened["mount"] as mount:
            assert mount.keys() == ["array", "empty", "table"]


def test_equivalent_openers_share_cache_identity_and_auth_does_not(caterva2_source, tmp_path):  # noqa: F811
    base, _, _, stats = caterva2_source
    cache = tmp_path / "cache"
    folders = []
    for token, source in [
        ("alice=secret", base + "@public/group"),
        ("alice=secret", blosc2.URLPath("@public/group", urlbase=base)),
        ("bob=secret", base + "@public/group"),
    ]:
        with blosc2.c2context(auth_token=token), blosc2.open(source, lazy=True, cache_dir=cache) as store:
            folders.append(store._owner.disk.path)
            with store["array"] as remote:
                remote[:1, :2]
    assert folders[0] == folders[1]
    assert folders[0] != folders[2]
    assert stats["fetches"] == 2


def test_hierarchy_summary_isolates_failed_sources(caterva2_source):  # noqa: F811
    base, _, _, stats = caterva2_source
    stats["groups"]["@public/mount"].append("broken")
    with blosc2.open(base + "@public") as store:
        summary = str(store.info)
        assert "listing incomplete" in summary
        assert "[unavailable]" in summary
        assert "array" in summary


def test_default_discovery_limits(caterva2_source):  # noqa: F811
    base, _, _, stats = caterva2_source
    stats["groups"]["@public"] = ["mount/array"] * 100001
    with blosc2.open(base + "@public") as store:
        assert store._owner.max_nodes == 10000
        with pytest.raises(ValueError, match="100000-entry"):
            store.keys()
