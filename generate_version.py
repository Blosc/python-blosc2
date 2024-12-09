# generate_version.py

import tomllib as toml

# Read the pyproject.toml file
with open("pyproject.toml", "rb") as f:
    pyproject = toml.load(f)

# Extract the version
version = pyproject["project"]["version"]

# Write the version to blosc2/_version.py
with open("src/blosc2/version.py", "w") as f:
    f.write(f'__version__ = "{version}"\n')
