# Requirements for developers

We are using Black as code formatter and Ruff as a linter.  These are automatically enforced
if you activate these as plugins for [pre-commit](https://pre-commit.com).  You can activate
the pre-commit actions by following the [instructions](https://pre-commit.com/#installation).
As the config files are already there, this essentially boils down to:

``` bash
  python -m pip install pre-commit
  pre-commit install
```

You are done!

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

We are using Sphinx for documentation.  You can build the documentation by executing

``` bash
  python -m sphinx doc html
```

You will find the documentation in the `html` directory.
