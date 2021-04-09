#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from __future__ import print_function

import os

from skbuild import setup


def cmake_bool(cond):
    return 'ON' if cond else 'OFF'


try:
    import cpuinfo
    cpu_info = cpuinfo.get_cpu_info()
except Exception:
    # newer cpuinfo versions fail to import on unsupported architectures
    cpu_info = None


# Blosc version
VERSION = open('VERSION').read().strip()
# Create the version.py file
open('blosc2/version.py', 'w').write('__version__ = "%s"\n' % VERSION)


setup(name="blosc2",
      setup_requires=[
          'cython>=0.29',
          'scikit-build',
          'pytest>=3.4.2'
      ],
      cmake_args=[
          '-DDEACTIVATE_SSE2:BOOL=%s' % cmake_bool(
              ('DISABLE_BLOSC_SSE2' in os.environ) or (cpu_info is None) or ('sse2' not in cpu_info['flags'])),
          '-DDEACTIVATE_AVX2:BOOL=%s' % cmake_bool(
              ('DISABLE_BLOSC_AVX2' in os.environ) or (cpu_info is None) or ('avx2' not in cpu_info['flags'])),
          '-DDEACTIVATE_LZ4:BOOL=%s' % cmake_bool(not int(os.environ.get('INCLUDE_LZ4', '1'))),
          # Snappy is disabled by default
          '-DDEACTIVATE_SNAPPY:BOOL=%s' % cmake_bool(not int(os.environ.get('INCLUDE_SNAPPY', '0'))),
          '-DDEACTIVATE_ZLIB:BOOL=%s' % cmake_bool(not int(os.environ.get('INCLUDE_ZLIB', '1'))),
          '-DDEACTIVATE_ZSTD:BOOL=%s' % cmake_bool(not int(os.environ.get('INCLUDE_ZSTD', '1'))),
      ],
      tests_require=['numpy', 'psutil'],
      packages=['blosc2'],
      package_dir={'blosc2': 'blosc2'},
      )
