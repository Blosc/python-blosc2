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