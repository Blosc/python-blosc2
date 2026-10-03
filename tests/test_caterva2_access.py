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
