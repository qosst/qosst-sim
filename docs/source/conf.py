# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from qosst_sim import __version__

sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "qosst-sim"
copyright = "2021-2024, Mayeul Chavanne, Yoann Piétri"
author = "Mayeul Chavanne, Yoann Piétri"

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx-prompt",
    "matplotlib.sphinxext.plot_directive",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

root_doc = "index"

latex_documents = [
    (
        root_doc,
        "qosst-sim.tex",
        "qosst-sim",
        author.replace(", ", "\\and ").replace(" and ", "\\and and "),
        "manual",
    ),
]

autoclass_content = "both"
autodoc_member_order = "bysource"

html_logo = "_static/qosst_logo_square_white.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}
myst_heading_anchors = 3

intersphinx_mapping = {
    "qosst-hal": ("https://qosst-hal.readthedocs.io/", None),
    "qosst-alice": ("https://qosst-alice.readthedocs.io/", None),
    "qosst-bob": ("https://qosst-bob.readthedocs.io/", None),
    "qosst-skr": ("https://qosst-skr.readthedocs.io/", None),
    "qosst-core": ("https://qosst-core.readthedocs.io/", None),
    "qosst": ("https://qosst.readthedocs.io/", None),
}
