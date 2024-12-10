import tomllib as toml

with open("pyproject.toml", "rb") as f:
    pyproject = toml.load(f)

version = pyproject["project"]["version"]

with open("src/blosc2/version.py", "w") as f:
    f.write(f'__version__ = "{version}"\n')
