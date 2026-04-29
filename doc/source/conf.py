# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

from recommonmark.transform import AutoStructify

sys.path.insert(0, os.path.abspath(".."))
import nexus_backend

# -- Project information -----------------------------------------------------

project = "nexus-backend"
copyright = "The Qibo team"
author = "The Qibo team"

# The full version, including alpha/beta/rc tags
release = nexus_backend.__version__


# -- General configuration ---------------------------------------------------
#
# https://stackoverflow.com/questions/56336234/build-fail-sphinx-error-contents-rst-not-found
master_doc = "index"

# Add any Sphinx extension module names here, as strings.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "recommonmark",
    "nbsphinx",
]

templates_path = ["_templates"]

source_suffix = {".rst": "restructuredtext", ".txt": "markdown", ".md": "markdown"}

autosectionlabel_prefix_document = True
enable_eval_rst = True

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"

html_title = "nexus-backend · v" + release

html_theme_options = {
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/qiboteam/nexus-backend/",
    "source_branch": "main",
    "source_directory": "doc/source/",
    "light_css_variables": {
        "color-brand-primary": "#6400FF",
        "color-brand-secondary": "#6400FF",
        "color-brand-content": "#6400FF",
    },
}

html_static_path = ["_static"]


# -- Intersphinx  -------------------------------------------------------------

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}


# -- Autodoc ------------------------------------------------------------------
autodoc_member_order = "bysource"


def setup(app):
    app.add_config_value("recommonmark_config", {"enable_eval_rst": True}, True)
    app.add_transform(AutoStructify)


html_show_sourcelink = False
