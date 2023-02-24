#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE.txt file in the root directory of this source tree)
#######################################################################

from textwrap import dedent

from skbuild import setup


def cmake_bool(cond):
    return "ON" if cond else "OFF"


# Blosc version
VERSION = open("VERSION").read().strip()
# Create the version.py file
open("blosc2/version.py", "w").write('__version__ = "%s"\n' % VERSION)

classifiers = dedent(
    """\
    Development Status :: 5 - Production/Stable
    Intended Audience :: Developers
    Intended Audience :: Information Technology
    Intended Audience :: Science/Research
    License :: OSI Approved :: BSD License
    Programming Language :: Python
    Topic :: Software Development :: Libraries :: Python Modules
    Operating System :: Microsoft :: Windows
    Operating System :: Unix
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.8
    Programming Language :: Python :: 3.9
    Programming Language :: Python :: 3.10
    Programming Language :: Python :: 3.11
    """
)


# read the contents of the README file
long_description = """
Python-Blosc2 is a Python package that wraps C-Blosc2, the newest version of
the Blosc compressor.  Python-Blosc2 reproduces quite well the API of its Python-Blosc
predecessor, so it can be used as a drop-in replacement (bar some exceptions; read the
docs if this is important for you).

In addition, Python-Blosc2 aims to leverage the new C-Blosc2 API so as to support
super-chunks, multi-dimensional arrays, serialization and other bells and whistles
introduced in C-Blosc2.

More info at: `https://www.blosc.org <https://www.blosc.org>`_
"""

setup(
    name="blosc2",
    version=VERSION,
    description="Python wrapper for the C-Blosc2 library.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    classifiers=[c for c in classifiers.split("\n") if c],
    python_requires=">=3.8, <4",
    author="Blosc Development Team",
    author_email="blosc@blosc.org",
    maintainer="Blosc Development Team",
    maintainer_email="blosc@blosc.org",
    url="https://github.com/Blosc/python-blosc2",
    license="https://opensource.org/licenses/BSD-3-Clause",
    platforms=["any"],
    packages=["blosc2"],
    package_dir={"blosc2": "blosc2"},
    install_requires=open("requirements-runtime.txt").read().split(),
)
