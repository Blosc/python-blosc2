#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE.txt file in the root directory of this source tree)
#######################################################################

from skbuild import setup

# Blosc version
VERSION = open("VERSION").read().strip()
# Create the version.py file
open("blosc2/version.py", "w").write(f'__version__ = "{VERSION}"\n')


def exclude_pkgconfig(cmake_manifest):
    """remove pkgconfig file from installation: gh-110."""
    return list(filter(lambda name: not (name.endswith(".pc")), cmake_manifest))


# These keywords need to be in setup()
# https://scikit-build.readthedocs.io/en/latest/usage.html#setuptools-options
setup(
    version=VERSION,
    packages=["blosc2"],
    package_dir={"blosc2": "blosc2"},
    include_package_data=True,
    cmake_process_manifest_hook=exclude_pkgconfig,
)
