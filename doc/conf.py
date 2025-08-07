# -- Path setup --------------------------------------------------------------
import os
import sys

import blosc2

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

hidden = "_ignore_multiple_size"


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None

    import importlib
    import inspect
    import os

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
