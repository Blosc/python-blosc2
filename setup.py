#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE.txt file in the root directory of this source tree)
#######################################################################

from skbuild import setup


def exclude_pkgconfig(cmake_manifest):
    """remove pkgconfig file from installation: gh-110."""
    return list(filter(lambda name: not (name.endswith(".pc")), cmake_manifest))


setup(
    packages=["blosc2"],
    cmake_process_manifest_hook=exclude_pkgconfig,
    # include_package_data=True,
)
