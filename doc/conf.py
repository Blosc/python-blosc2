# -- Path setup --------------------------------------------------------------
import os
import sys

import blosc2

sys.path.insert(0,  os.path.abspath(os.path.dirname(blosc2.__file__)))

project = 'Python-Blosc2'
copyright = '2019-present, The Blosc Developers'
author = 'The Blosc Developers'
extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'myst_parser'
]
source_suffix = [".rst", ".md"]
html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
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
    "github_url": "https://github.com/Blosc/c-blosc2",
    "twitter_url": "https://twitter.com/Blosc2",
}

html_show_sourcelink = False
