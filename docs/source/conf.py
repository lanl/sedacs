# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://gitlab.lanl.gov/sedacs/sedacs

project = 'SEDACS'
copyright = '2025, SEDACS team'
author = 'SEDACS team'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#extensions = ['sphinx.ext.mathjax', 'sphinx.ext.autodoc', 'myst_parser', 'sphinx.ext.autodoc', 'autoapi.extension']
extensions = ['sphinx.ext.mathjax', 'sphinx.ext.autodoc', 'sphinx_mdinclude', 'autoapi.extension']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autoapi_type = 'python'
autoapi_dirs = '../../src/sedacs'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'press'
html_static_path = ['_static']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown'
#    '.md': 'markdown',
}
