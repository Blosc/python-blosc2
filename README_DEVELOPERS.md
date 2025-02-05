# Requirements for developers

We are using Ruff as code formatter and as a linter.  It is automatically enforced
if you activate these as plugins for [pre-commit](https://pre-commit.com).  You can activate
the pre-commit actions by following the [instructions](https://pre-commit.com/#installation).
As the config files are already there, this essentially boils down to:

``` bash
  python -m pip install pre-commit
  pre-commit install
```

You are done!

## Building from sources

``python-blosc2`` includes the C-Blosc2 source code and can be built in place:

``` bash
    git clone https://github.com/Blosc/python-blosc2/
    cd python-blosc2
    pip install .   # add -e for editable mode
```

That's it! You can now proceed to the testing section.

## Testing

We are using pytest for testing.  You can run the tests by executing

``` bash
  pytest
```

If you want to run a heavyweight version of the tests, you can use the following command:

``` bash
  pytest -m "heavy"
```

If you want to run the network tests, you can use the following command:

``` bash
  pytest -m "network"
```

## Documentation

We are using Sphinx for documentation.  You can build the documentation by executing:

``` bash
  cd doc
  rm -rf html _build
  python -m sphinx . html
```
[You may need to install the `pandoc` package first: https://pandoc.org/installing.html]

You will find the documentation in the `html` directory.
