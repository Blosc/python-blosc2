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
    "numpydoc",
    "myst_parser",
    "sphinx_paramlinks",
    "sphinx_panels",
    "nbsphinx",
]
source_suffix = [".rst", ".md"]
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
html_logo = "_static/blosc-logo_256.png"
html_favicon = "_static/blosc-logo_128.png"
html_theme_options = {
    "logo": {
        "link": "/index",
        "alt_text": "Blosc",
    },
    "external_links": [
        {"name": "C-Blosc2", "url": "/c-blosc2/c-blosc2.html"},
        {"name": "Python-Blosc", "url": "/python-blosc/python-blosc.html"},
        {"name": "Blosc In Depth", "url": "/pages/blosc-in-depth/"},
        {"name": "Donate to Blosc", "url": "/pages/donate/"},
    ],
    "github_url": "https://github.com/Blosc/python-blosc2",
    "twitter_url": "https://twitter.com/Blosc2",
}

html_show_sourcelink = False

autosummary_generate_overwrite = False

hidden = "_ignore_multiple_size"


def process_sig(app, what, name, obj, options, signature, return_annotation):
    if signature and hidden in signature:
        signature = signature.split(hidden)[0] + ")"
    return (signature, return_annotation)


def setup(app):
    app.connect("autodoc-process-signature", process_sig)
