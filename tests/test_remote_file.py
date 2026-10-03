"""Deterministic Caterva2 ordinary-file byte transport and lifetime tests."""

import pytest
from test_remote_caterva2 import caterva2_source  # noqa: F401

import blosc2


@pytest.fixture
def file_source(caterva2_source):  # noqa: F811
    base, _, _, stats = caterva2_source
    payload = b"# Hello\n\nAn ordinary **Markdown** file.\n" * 20
    stream = blosc2.SChunk(chunksize=64, cparams={"typesize": 1})
    for offset in range(0, len(payload), 64):
        stream.append_data(payload[offset : offset + 64])
    stats["files"]["@public/README.md"] = stream
    stats["groups"]["@public"].append("README.md")
    return base, payload, stats


def test_discovery_ranges_and_download(file_source, tmp_path):
    base, payload, stats = file_source
    with blosc2.open(base + "@public") as store:
        assert store.kind("README.md") == "file"
        assert not stats["file_chunks"]
        with store["README.md"] as file:
            assert isinstance(file, blosc2.RemoteFile)
            assert file.name == "README.md"
            assert file.attrs["title"] == "ordinary file"
            assert file.read_bytes(59, 135) == payload[59:135]
            assert file.read_bytes(1, 1) == b""
            for start, stop in [(-1, 3), (4, 3), (True, 4), (0, len(payload) + 1)]:
                with pytest.raises(ValueError):
                    file.read_bytes(start, stop)
            dest = tmp_path / "README.md"
            progress = []
            file.download(dest, progress=lambda done, total: progress.append((done, total)))
            assert dest.read_bytes() == payload
            assert progress[-1] == (len(payload), len(payload))
            with pytest.raises(FileExistsError):
                file.download(dest)
            with pytest.raises(NotImplementedError):
                file.save(tmp_path / "file.b2z")


@pytest.mark.parametrize(
    "policy", [blosc2.CachePolicy.NONE, blosc2.CachePolicy.MEMORY, blosc2.CachePolicy.DISK]
)
def test_cache_and_child_lifetime(file_source, tmp_path, policy):
    base, payload, stats = file_source
    options = {"cache_policy": policy}
    if policy is blosc2.CachePolicy.DISK:
        options["cache_dir"] = tmp_path / "cache"
    store = blosc2.open(base + "@public", **options)
    file = store["README.md"]
    store.close()
    with file:
        assert file.read_bytes(0, 10) == payload[:10]
        before = len(stats["file_chunks"])
        assert file.read_bytes(0, 10) == payload[:10]
        assert len(stats["file_chunks"]) == before + (policy is blosc2.CachePolicy.NONE)
    with pytest.raises(RuntimeError, match="closed"):
        file.read_bytes(0, 1)
    if policy is blosc2.CachePolicy.DISK:
        with blosc2.open(base + "@public/README.md", **options) as direct:
            assert direct.read_bytes(0, 10) == payload[:10]
        # A different selected source has a distinct cache identity.
        with blosc2.open(base + "@public", **options) as root, root["README.md"] as warm:
            before = len(stats["file_chunks"])
            warm.read_bytes(0, 10)
            assert len(stats["file_chunks"]) == before


def test_direct_file_and_cancelled_atomic_download(file_source, tmp_path):
    base, payload, _ = file_source
    dest = tmp_path / "test.md"
    dest.write_bytes(b"old data")
    with blosc2.open(base + "@public/README.md") as file:
        assert file.read_bytes() == payload
        with pytest.raises(InterruptedError):
            file.download(dest, overwrite=True, cancel=lambda: True)
        assert dest.read_bytes() == b"old data"
        assert not list(tmp_path.glob(".b2view-download-*"))
        with pytest.raises(NotImplementedError, match="lazy=True"):
            blosc2.open(blosc2.URLPath("@public/README.md", urlbase=base))


def test_metadata_validation(file_source):
    base, _, stats = file_source
    for overrides in ({"nbytes": True}, {"chunksize": 0}, {"nchunks": 999}, {"cparams": {"typesize": 0}}):
        stats["file_metadata"]["@public/README.md"] = overrides
        with pytest.raises(ValueError):
            blosc2.open(base + "@public/README.md")
