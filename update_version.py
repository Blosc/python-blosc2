#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import re
import sys


def update_version(new_version):
    # Update version in pyproject.toml
    with open("pyproject.toml") as file:
        pyproject_content = file.read()
    pyproject_content = re.sub(r'version = ".*"', f'version = "{new_version}"', pyproject_content)
    with open("pyproject.toml", "w") as file:
        file.write(pyproject_content)

    # Update version in src/blosc2/version.py
    with open("src/blosc2/version.py") as file:
        version_content = file.read()
    version_content = re.sub(r'__version__ = ".*"', f'__version__ = "{new_version}"', version_content)
    with open("src/blosc2/version.py", "w") as file:
        file.write(version_content)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <new_version>")
        sys.exit(1)
    new_version = sys.argv[1]
    update_version(new_version)
    print(f"Version updated to {new_version}")
