#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Unit tests for b2view's CLI source resolution (no app session needed)."""

from __future__ import annotations

import pytest

from blosc2.b2view.cli import (
    DEFAULT_DOWNLOAD_PATH,
    DOWNLOAD_BASE_URL,
    INFO_BASE_URL,
    main,
    resolve_source,
)


def test_positional_path_is_used_as_is():
    assert resolve_source("local.b2z", None) == ("local.b2z", None, None)


def test_download_default_path_when_missing():
    urlpath, url, info_url = resolve_source(None, DEFAULT_DOWNLOAD_PATH, exists=lambda p: False)
    # The bundle is saved in the cwd under its basename...
    assert urlpath == "chicago-taxi-flat.b2z"
    # ...but the URLs use the full @public-relative path.
    assert url == DOWNLOAD_BASE_URL + DEFAULT_DOWNLOAD_PATH
    assert info_url == INFO_BASE_URL + DEFAULT_DOWNLOAD_PATH


def test_download_skipped_when_file_already_in_cwd():
    urlpath, url, info_url = resolve_source(None, "large/foo.b2z", exists=lambda p: True)
    assert urlpath == "foo.b2z"  # basename only
    assert url is None  # present locally -> no fetch
    assert info_url is None


def test_download_url_dest_is_basename():
    urlpath, url, info_url = resolve_source(None, "sub/dir/bundle.b2z", exists=lambda p: False)
    assert urlpath == "bundle.b2z"
    assert url == DOWNLOAD_BASE_URL + "sub/dir/bundle.b2z"
    assert info_url == INFO_BASE_URL + "sub/dir/bundle.b2z"


def test_download_and_positional_exclusive():
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_source("local.b2z", "foo.b2z")


def test_no_source_is_an_error():
    with pytest.raises(ValueError, match="provide a path"):
        resolve_source(None, None)


@pytest.mark.parametrize("options", [{}, {"profile": "blosc2", "endpoint_url": "https://s3.example.com"}])
def test_remote_options_reach_browser(monkeypatch, options):
    pytest.importorskip("textual")
    from blosc2.b2view import app as app_module
    from blosc2.b2view.app import B2ViewApp

    url = "s3://blosc2/hierarchy.b2z/d0/d1/a2"
    opened = []

    def open_browser(path, **kwargs):
        opened.append((path, kwargs))
        raise LookupError("Stop before remote I/O")

    monkeypatch.setattr(app_module, "StoreBrowser", open_browser)

    def run(app, **kwargs):
        monkeypatch.setattr(app, "_deliver_remote", lambda *args: None)
        # Exercise the worker body without starting a terminal or a thread.
        app._open_remote.__wrapped__(app, app._remote_session, "/")

    monkeypatch.setattr(B2ViewApp, "run", run)
    argv = [url]
    for key, value in options.items():
        argv.extend(["--" + key.replace("_", "-"), value])
    assert main(argv) == 0
    assert opened == [(url, {"storage_options": options or None, "cache_dir": None})]


def test_cache_dir_reaches_browser(monkeypatch, tmp_path):
    pytest.importorskip("textual")
    from blosc2.b2view import app as app_module
    from blosc2.b2view.app import B2ViewApp

    url = "s3://blosc2/hierarchy.b2z"
    opened = []

    def open_browser(path, **kwargs):
        opened.append((path, kwargs))
        raise LookupError("Stop before remote I/O")

    monkeypatch.setattr(app_module, "StoreBrowser", open_browser)

    def run(app, **kwargs):
        monkeypatch.setattr(app, "_deliver_remote", lambda *args: None)
        app._open_remote.__wrapped__(app, app._remote_session, "/")

    monkeypatch.setattr(B2ViewApp, "run", run)
    cache_path = str(tmp_path / "cache")
    assert main([url, "--cache-dir", cache_path]) == 0
    assert opened == [(url, {"storage_options": None, "cache_dir": cache_path})]


def test_max_cache_bytes_reaches_browser(monkeypatch):
    pytest.importorskip("textual")
    from blosc2.b2view import app as app_module
    from blosc2.b2view.app import B2ViewApp

    url = "s3://blosc2/hierarchy.b2z"
    opened = []

    def open_browser(path, **kwargs):
        opened.append((path, kwargs))
        raise LookupError("Stop before remote I/O")

    monkeypatch.setattr(app_module, "StoreBrowser", open_browser)

    def run(app, **kwargs):
        monkeypatch.setattr(app, "_deliver_remote", lambda *args: None)
        app._open_remote.__wrapped__(app, app._remote_session, "/")

    monkeypatch.setattr(B2ViewApp, "run", run)
    assert main([url, "--max-cache-bytes", "1048576"]) == 0
    assert opened == [(url, {"storage_options": None, "cache_dir": None, "max_cache_bytes": 1048576})]
