#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from __future__ import print_function

import os
from textwrap import dedent
from skbuild import setup


def cmake_bool(cond):
    return 'ON' if cond else 'OFF'


try:
    import cpuinfo
    cpu_info = cpuinfo.get_cpu_info()
except:
    # newer cpuinfo versions fail to import on unsupported architectures
    cpu_info = None


# Blosc version
VERSION = open('VERSION').read().strip()
# Create the version.py file
open('blosc2/version.py', 'w').write('__version__ = "%s"\n' % VERSION)

classifiers = dedent("""\
    Development Status :: 3 - Alpha
    Intended Audience :: Developers
    Intended Audience :: Information Technology
    Intended Audience :: Science/Research
    License :: OSI Approved :: BSD License
    Programming Language :: Python
    Topic :: Software Development :: Libraries :: Python Modules
    Operating System :: Microsoft :: Windows
    Operating System :: Unix
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.6
    Programming Language :: Python :: 3.7
    Programming Language :: Python :: 3.8
    """)

setup(
    name="blosc2",
    version=VERSION,
    description='Python wrapper for the C-Blosc2 library.',
    long_description=None,
    classifiers=[c for c in classifiers.split("\n") if c],
    author='Blosc Development Team',
    author_email='blosc@blosc.org',
    maintainer='Blosc Development Team',
    maintainer_email='blosc@blosc.org',
    url='https://github.com/Blosc/cat4py',
    license='https://opensource.org/licenses/BSD-3-Clause',
    platforms=['any'],
    setup_requires=[
        'cython>=0.29',
        'scikit-build',
        'py-cpuinfo'
    ],
    cmake_args=[
        '-DDEACTIVATE_SSE2:BOOL=%s' % cmake_bool(
          ('DISABLE_BLOSC_SSE2' in os.environ) or (cpu_info is None) or ('sse2' not in cpu_info['flags'])),
        '-DDEACTIVATE_AVX2:BOOL=%s' % cmake_bool(
          ('DISABLE_BLOSC_AVX2' in os.environ) or (cpu_info is None) or ('avx2' not in cpu_info['flags'])),
        '-DDEACTIVATE_LZ4:BOOL=%s' % cmake_bool(not int(os.environ.get('INCLUDE_LZ4', '1'))),
        '-DDEACTIVATE_ZLIB:BOOL=%s' % cmake_bool(not int(os.environ.get('INCLUDE_ZLIB', '1'))),
        '-DDEACTIVATE_ZSTD:BOOL=%s' % cmake_bool(not int(os.environ.get('INCLUDE_ZSTD', '1'))),
    ],
    tests_require=['numpy', 'psutil'],
    packages=['blosc2'],
    package_dir={'blosc2': 'blosc2'},
)
