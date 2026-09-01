# -- Path setup --------------------------------------------------------------
import importlib
import inspect
import os
import re
import sys
from pathlib import Path

import numpy as np
from sphinx.util import logging as sphinx_logging

import blosc2
from blosc2.utils import elementwise_funcs, reducers


def genbody(f, func_list, lib="blosc2"):
    for func in func_list:
        f.write(f"    {func}\n")

    f.write("\n\n\n")
    for func in func_list:
        f.write(f".. autofunction:: {lib}.{func}\n")


sys.path.insert(0, os.path.abspath(os.path.dirname(blosc2.__file__)))

logger = sphinx_logging.getLogger(__name__)

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
    "sphinx_reredirects",
    # For some reason, the following extensions are not working
    # "IPython.sphinxext.ipython_directive",
    # "IPython.sphinxext.ipython_console_highlighting",
]
source_suffix = [".rst", ".md"]
# Redirect stubs for pages that moved out of getting_started/ (their old URLs
# are linked from released READMEs on PyPI and from blog posts).
redirects = {
    "getting_started/b2view": "../guides/b2view.html",
    "getting_started/parquet_to_blosc2": "../guides/parquet_to_blosc2.html",
    "getting_started/sharing_across_processes": "../guides/sharing_across_processes.html",
    "getting_started/dsl_syntax": "../reference/dsl_syntax.html",
    "getting_started/tutorials": "../tutorials/index.html",
    "getting_started/tutorials/01.ndarray-basics": "../../tutorials/01.ndarray-basics.html",
    "getting_started/tutorials/02.lazyarray-expressions": "../../tutorials/02.lazyarray-expressions.html",
    "getting_started/tutorials/03.lazyarray-udf": "../../tutorials/03.lazyarray-udf.html",
    "getting_started/tutorials/03.lazyarray-udf-kernels": "../../tutorials/03.lazyarray-udf-kernels.html",
    "getting_started/tutorials/04.reductions": "../../tutorials/04.reductions.html",
    "getting_started/tutorials/05.persistent-reductions": "../../tutorials/05.persistent-reductions.html",
    "getting_started/tutorials/06.remote_proxy": "../../tutorials/06.remote_proxy.html",
    "getting_started/tutorials/07.schunk-basics": "../../tutorials/07.schunk-basics.html",
    "getting_started/tutorials/08.schunk-slicing_and_beyond": "../../tutorials/08.schunk-slicing_and_beyond.html",
    "getting_started/tutorials/09.ucodecs-ufilters": "../../tutorials/09.ucodecs-ufilters.html",
    "getting_started/tutorials/10.prefilters": "../../tutorials/10.prefilters.html",
    "getting_started/tutorials/11.containers": "../../tutorials/11.containers.html",
    "getting_started/tutorials/11.objectarray": "../../tutorials/11.objectarray.html",
    "getting_started/tutorials/12.batcharray": "../../tutorials/12.batcharray.html",
    "getting_started/tutorials/13.ctable-basics": "../../tutorials/13.ctable-basics.html",
    "getting_started/tutorials/14.indexing-arrays": "../../tutorials/14.indexing-arrays.html",
    "getting_started/tutorials/15.indexing-ctables": "../../tutorials/15.indexing-ctables.html",
}
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

exclude_patterns = ["_build", ".DS_Store", "**.ipynb_checkpoints", "tutorials/images/**"]

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

Note: The functions ``real``, ``imag``, ``contains``, ``where`` are not technically ufuncs.

.. currentmodule:: blosc2

.. autosummary::

"""
    )
    genbody(f, blosc2_ufuncs)

# GENERATE additional_funcs.rst
blosc2_addfuncs = sorted(set(elementwise_funcs) - set(blosc2_ufuncs))
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
        """

Type Utilities
--------------

The following functions are useful for working with datatypes.

.. currentmodule:: blosc2

.. autosummary::

"""
    )
    genbody(f, blosc2_dtypefuncs)

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
    genbody(f, sorted(linalg_funcs), "blosc2.linalg")

with open("reference/reduction_functions.rst", "w") as f:
    f.write(
        """Reduction Functions
-------------------

Contrarily to lazy functions, reduction functions are evaluated eagerly, and the result is always a NumPy array (although this can be converted internally into an :ref:`NDArray <NDArray>` if you pass any :func:`blosc2.empty` arguments in ``kwargs``).

Reduction operations can be used with any of :ref:`NDArray <NDArray>`, :ref:`C2Array <C2Array>`, :ref:`NDField <NDField>` and :ref:`LazyExpr <LazyExpr>`. Again, although these can be part of a :ref:`LazyExpr <LazyExpr>`, you must be aware that they are not lazy, but will be evaluated eagerly during the construction of a LazyExpr instance (this might change in the future). When the input is a :ref:`LazyExpr`, reductions accept ``fp_accuracy`` to control floating-point accuracy, and it is forwarded to :func:`LazyExpr.compute`.

.. currentmodule:: blosc2

.. autosummary::

"""
    )
    genbody(f, sorted(reducers))
    f.write(
        """
Grouped reductions
~~~~~~~~~~~~~~~~~~

The :func:`blosc2.group_reduce` function is a lower-level, array-oriented primitive that groups one-dimensional keys and applies eager reductions to the associated values.

.. autofunction:: blosc2.group_reduce
"""
    )

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

    github_base_url = "https://github.com/Blosc/python-blosc2/blob/main/"
    fn = os.path.abspath(fn)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        relpath = os.path.relpath(fn, start=repo_root)
    except ValueError:
        relpath = None
    if relpath is None or relpath.startswith(".."):
        # Release docs may be built from an installed wheel/sdist, where source
        # files live under site-packages/blosc2 instead of the repository's
        # src/blosc2 tree.  Map those installed package paths back to the repo.
        package_root = os.path.abspath(os.path.dirname(blosc2.__file__))
        try:
            package_relpath = os.path.relpath(fn, start=package_root)
        except ValueError:
            return None
        if package_relpath.startswith(".."):
            return None
        relpath = os.path.join("src", "blosc2", package_relpath)

    return f"{github_base_url}{relpath}#L{lineno}"


def process_sig(app, what, name, obj, options, signature, return_annotation):
    if signature and hidden in signature:
        signature = signature.split(hidden)[0] + ")"
    return (signature, return_annotation)


# -- Undocumented public API tripwire ------------------------------------------
#
# Every name in ``blosc2.__all__`` should either be documented on some reference
# page or be listed in ``undocumented_members`` below.  We check that at the end
# of the build and warn about the difference, so a newly added public object
# cannot slip in without someone deciding where it belongs.  Build the docs with
# ``-W`` (as CI does) to turn that warning into a failure.

_AUTODOC_DIRECTIVE = re.compile(
    r"^\s*\.\.\s+(?:autoclass|autofunction|autodata|autoexception|autodecorator)::"
    r"\s*([\w.]+)"
)
_CURRENTMODULE = re.compile(r"^\s*\.\.\s+(?:currentmodule|module)::\s*([\w.]+)")
_AUTOSUMMARY = re.compile(r"^(\s*)\.\.\s+autosummary::")
_AUTOSUMMARY_ENTRY = re.compile(r"^\s*~?([\w.]+)\s*$")
_AUTOSUMMARY_OPTION = re.compile(r"^\s*:[\w-]+:")

# Public members deliberately left undocumented, so that the check below only
# ever flags genuine omissions.  Trimming this set is a standing invitation.
undocumented_members = {
    # Array-API constants, self-explanatory and documented upstream.
    "e",
    "inf",
    "nan",
    "newaxis",
    "pi",
    "DEFAULT_COMPLEX",
    "DEFAULT_FLOAT",
    "DEFAULT_INDEX",
    "DEFAULT_INT",
    "DEFAULT_NULL_POLICY",
    "DSLKernel",
    "DictionarySpec",
    "LazyUDF",
    "NDArraySpec",
    "are_partitions_aligned",
    "are_partitions_behaved",
    "array_from_ffi_ptr",
    "as_simpleproxy",
    "get_cpu_info",
    "linalg_funcs_list",
}

documented_members = set()


def _in_blosc2(dotted_name):
    """True if ``dotted_name`` names an attribute of ``blosc2`` or a submodule."""
    parent, _, attr = dotted_name.rpartition(".")
    if parent == "blosc2":
        return attr
    if not parent.startswith("blosc2."):
        return None
    try:
        importlib.import_module(parent)
    except ImportError:
        return None
    return attr


def collect_documented_members(docdir):
    """Names carrying an autodoc directive somewhere under ``docdir``."""
    documented = set()
    for path in sorted(Path(docdir).rglob("*.rst")):
        module = None
        summary_indent = None
        for line in path.read_text(encoding="utf-8").splitlines():
            match = _CURRENTMODULE.match(line)
            if match:
                module, summary_indent = match.group(1), None
                continue
            match = _AUTOSUMMARY.match(line)
            if match:
                summary_indent = len(match.group(1))
                continue
            match = _AUTODOC_DIRECTIVE.match(line)
            if match:
                summary_indent, name = None, match.group(1)
            elif summary_indent is not None:
                # Inside an autosummary block: one bare (possibly dotted) name
                # per line, more indented than the directive itself.
                if not line.strip() or _AUTOSUMMARY_OPTION.match(line):
                    continue
                entry = _AUTOSUMMARY_ENTRY.match(line)
                if entry is None or len(line) - len(line.lstrip()) <= summary_indent:
                    summary_indent = None
                    continue
                name = entry.group(1)
            else:
                continue
            if "." not in name:
                if module is None:
                    continue
                name = f"{module}.{name}"
            attr = _in_blosc2(name)
            if attr:
                documented.add(attr)
    return documented


def gather_documented(app):
    documented_members.update(collect_documented_members(app.srcdir))


def check_undocumented(app, exception):
    """Warn about public ``blosc2`` names that no reference page documents."""
    if exception is not None:
        return
    unclassified = set(blosc2.__all__) - documented_members - undocumented_members
    if unclassified:
        logger.warning(
            "these public blosc2 members are not documented on any reference "
            "page: %s.  Add them to the appropriate page under doc/reference/, "
            "or to undocumented_members in doc/conf.py.",
            ", ".join(sorted(unclassified)),
        )


def setup(app):
    app.connect("autodoc-process-signature", process_sig)
    app.connect("builder-inited", gather_documented)
    app.connect("build-finished", check_undocumented)


# Allow errors (e.g. with numba asking for a specific numpy version)
nbsphinx_allow_errors = True
