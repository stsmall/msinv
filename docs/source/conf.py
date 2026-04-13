"""Sphinx configuration for msinv documentation."""
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information ----------------------------------------------
project = 'msinv'
copyright = '2026, Scott T. Small'
author = 'Scott T. Small'
release = '0.1.0'
version = '0.1.0'

# -- General configuration --------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',       # numpy/google docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_parser',               # markdown support
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output ------------------------------------------
html_theme = 'furo'                # modern, clean theme
html_static_path = ['_static']
html_title = 'msinv documentation'

# -- Autodoc settings -------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}
autosummary_generate = True

# -- Intersphinx mapping ----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'msprime': ('https://tskit.dev/msprime/docs/stable/', None),
    'tskit': ('https://tskit.dev/tskit/docs/stable/', None),
}

# -- Napoleon settings ------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
