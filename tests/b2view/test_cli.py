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


def test_download_urls_keep_relative_path_dest_is_basename():
    urlpath, url, info_url = resolve_source(None, "sub/dir/bundle.b2z", exists=lambda p: False)
    assert urlpath == "bundle.b2z"
    assert url == DOWNLOAD_BASE_URL + "sub/dir/bundle.b2z"
    assert info_url == INFO_BASE_URL + "sub/dir/bundle.b2z"


def test_download_and_positional_are_mutually_exclusive():
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_source("local.b2z", "foo.b2z")


def test_no_source_is_an_error():
    with pytest.raises(ValueError, match="provide a path"):
        resolve_source(None, None)
