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
