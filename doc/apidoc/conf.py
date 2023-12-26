# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../gsm'))
sys.path.insert(0, os.path.abspath('../../gsm/data'))
sys.path.insert(0, os.path.abspath('../../gsm/models'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GSM'
copyright = '2023, Max V'
author = 'Max V'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#extensions = ['sphinx.ext.autodoc']
extensions = ['sphinx_automodapi.automodapi','sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = '.rst'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_css_files = ['css/custom.css']


def patch_automodapi(app):
    """Monkey-patch the automodapi extension to exclude imported members"""
    from sphinx_automodapi import automodsumm
    from sphinx_automodapi.utils import find_mod_objs
    automodsumm.find_mod_objs = lambda *args: find_mod_objs(args[0], onlylocals=True)

def setup(app):
    app.connect("builder-inited", patch_automodapi)