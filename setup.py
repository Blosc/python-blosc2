#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE.txt file in the root directory of this source tree)
#######################################################################

from skbuild import setup


def cmake_bool(cond):
    return "ON" if cond else "OFF"


# Blosc version
VERSION = open("VERSION").read().strip()
# Create the version.py file
open("blosc2/version.py", "w").write('__version__ = "%s"\n' % VERSION)

# These keywords need to be in setup()
# https://scikit-build.readthedocs.io/en/latest/usage.html#setuptools-options
setup(
    version=VERSION,
    packages=["blosc2"],
    package_dir={"blosc2": "blosc2"},
    include_package_data=True,
    install_requires=open("requirements-runtime.txt").read().split(),
)
