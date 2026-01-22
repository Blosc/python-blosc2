# -- Path setup --------------------------------------------------------------
import inspect
import os
import sys

import numpy as np

import blosc2


def genbody(f, func_list, lib="blosc2"):
    for func in func_list:
        f.write(f"    {func}\n")

    f.write("\n\n\n")
    for func in func_list:
        f.write(f".. autofunction:: {lib}.{func}\n")


sys.path.insert(0, os.path.abspath(os.path.dirname(blosc2.__file__)))

project = "Python-Blosc2"
copyright = "2019-present, The Blosc Developers"
author = "The Blosc Developers"
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "numpydoc",
    "myst_parser",
    "sphinx_paramlinks",
    "sphinx_design",
    "nbsphinx",
    # For some reason, the following extensions are not working
    # "IPython.sphinxext.ipython_directive",
    # "IPython.sphinxext.ipython_console_highlighting",
]
source_suffix = [".rst", ".md"]
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
]
html_logo = "_static/blosc-logo_256.png"
# Just use the favicon from the parent project
# html_favicon = "_static/blosc-logo_128.png"
html_favicon = "_static/blosc-favicon_64x64.png"
html_theme_options = {
    "logo": {
        "link": "/index",
        "alt_text": "Blosc",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Blosc/python-blosc2",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Mastodon",
            "url": "https://fosstodon.org/@Blosc2",
            "icon": "fab fa-mastodon",
        },
        {
            "name": "Bluesky",
            "url": "https://bsky.app/profile/blosc.org",
            "icon": "fas fa-cloud-sun",
        },
    ],
    "external_links": [
        {"name": "C-Blosc2", "url": "/c-blosc2/c-blosc2.html"},
        {"name": "Python-Blosc2", "url": "/python-blosc2/"},
        {"name": "Donate to Blosc", "url": "/pages/donate/"},
    ],
}

exclude_patterns = ["_build", ".DS_Store", "**.ipynb_checkpoints"]

html_show_sourcelink = False

autosummary_generate_overwrite = False
autosummary_generate = True

# GENERATE ufuncs.rst
blosc2_ufuncs = []
for name, obj in vars(np).items():
    if isinstance(obj, np.ufunc) and hasattr(blosc2, name):
        blosc2_ufuncs.append(name)

with open("reference/ufuncs.rst", "w") as f:
    f.write(
        """Universal Functions (`ufuncs`)
------------------------------

The following elementwise functions can be used for computing with any of :ref:`NDArray <NDArray>`, :ref:`C2Array <C2Array>`, :ref:`NDField <NDField>` and :ref:`LazyExpr <LazyExpr>`.

Their result is always a :ref:`LazyExpr` instance, which can be evaluated (with ``compute`` or ``__getitem__``) to get the actual values of the computation.

Note: The functions ``conj``, ``real``, ``imag``, ``contains``, ``where`` are not technically ufuncs.

.. currentmodule:: blosc2

.. autosummary::

"""
    )
    genbody(f, blosc2_ufuncs)

# GENERATE additional_funcs.rst
blosc2_addfuncs = sorted(["conj", "real", "imag", "contains", "where", "clip", "round"])
blosc2_dtypefuncs = sorted(["astype", "can_cast", "result_type", "isdtype"])

with open("reference/additional_funcs.rst", "w") as f:
    f.write(
        """Additional Functions and Type Utilities
=======================================

Functions
---------

The following functions can also be used for computing with any of :ref:`NDArray <NDArray>`, :ref:`C2Array <C2Array>`, :ref:`NDField <NDField>` and :ref:`LazyExpr <LazyExpr>`.

Their result is typically a :ref:`LazyExpr` instance, which can be evaluated (with ``compute`` or ``__getitem__``) to get the actual values of the computation.

.. currentmodule:: blosc2

.. autosummary::

"""
    )
    genbody(f, blosc2_addfuncs)
    f.write(
        """Type Utilities
--------------

The following functions are useful for working with datatypes.

.. currentmodule:: blosc2

.. autosummary::

"""
    )
    genbody(f, blosc2_dtypefuncs)

# GENERATE index_funcs.rst
blosc2_indexfuncs = sorted(
    [
        "count_nonzero",
        "squeeze",
        "expand_dims",
        "sort",
        "take",
        "take_along_axis",
        "broadcast_to",
        "meshgrid",
        "indices",
    ]
)

with open("reference/index_funcs.rst", "w") as f:
    f.write(
        """Indexing Functions and Utilities
=======================================

The following functions are useful for performing indexing and other associated operations.

.. currentmodule:: blosc2

.. autosummary::

"""
    )
    genbody(f, blosc2_indexfuncs)

# GENERATE linear_algebra.rst
linalg_funcs = [
    name
    for name, obj in vars(blosc2.linalg).items()
    if (inspect.isfunction(obj) and getattr(obj, "__doc__", None))
]

with open("reference/linalg.rst", "w") as f:
    f.write(
        """Linear Algebra
-----------------
The following functions can be used for computing linear algebra operations with :ref:`NDArray <NDArray>`.

.. currentmodule:: blosc2.linalg

.. autosummary::

"""
    )
    genbody(f, linalg_funcs, "blosc2.linalg")

hidden = "_ignore_multiple_size"


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None

    import importlib
    import inspect

    # Modify this to point to your package
    module_name = info["module"]
    full_name = info["fullname"]

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None

    obj = module
    for part in full_name.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None

    try:
        fn = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None

    # Replace this with your repo info
    github_base_url = "https://github.com/Blosc/python-blosc2/blob/main/"
    # Get the path relative to the repository root, not the module directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(blosc2.__file__), "..", ".."))
    relpath = os.path.relpath(fn, start=repo_root)
    return f"{github_base_url}{relpath}#L{lineno}"


def process_sig(app, what, name, obj, options, signature, return_annotation):
    if signature and hidden in signature:
        signature = signature.split(hidden)[0] + ")"
    return (signature, return_annotation)


def setup(app):
    app.connect("autodoc-process-signature", process_sig)


# Allow errors (e.g. with numba asking for a specific numpy version)
nbsphinx_allow_errors = True
