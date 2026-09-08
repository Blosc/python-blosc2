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
def test_remote_options_reach_open(monkeypatch, options):
    pytest.importorskip("textual")
    import blosc2
    from blosc2.b2view.app import B2ViewApp

    url = "s3://blosc2/hierarchy.b2z/d0/d1/a2"
    opened = []
    monkeypatch.setattr(blosc2, "open", lambda path, **kwargs: opened.append((path, kwargs)))

    def run(app, **kwargs):
        # Stop after the real startup opens its browser, before widget lookup.
        def no_widgets(*args):
            raise LookupError("No widgets in CLI test")

        monkeypatch.setattr(app, "query_one", no_widgets)
        with pytest.raises(LookupError, match="No widgets"):
            app._start_browsing()

    monkeypatch.setattr(B2ViewApp, "run", run)
    argv = [url]
    for key, value in options.items():
        argv.extend(["--" + key.replace("_", "-"), value])
    assert main(argv) == 0
    expected = {"mode": "r", "lazy": True}
    if options:
        expected["storage_options"] = options
    assert opened == [(url, expected)]
