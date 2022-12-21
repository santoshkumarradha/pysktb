# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.append(os.path.abspath("../../"))

project = "PySKTB"
copyright = "2022, Santosh Kumar Radha"
author = "Santosh Kumar Radha"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "autodocsumm",
    "nbsphinx",
    "sphinx_wagtail_theme",
    "sphinx_copybutton",
]

autosummary_generate = True
autodoc_default_options = {"autosummary": True}
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_wagtail_theme"
project = "PySKTB"
html_theme_options = dict(
    project_name=project,
    logo="logo.png",
    logo_alt="PySKTB",
    logo_height=70,
    logo_url="/",
    logo_width=45,
    github_url="https://github.com/santoshkumarradha/pysktb",
)
html_static_path = ["_static"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
nbsphinx_execute = "never"
highlight_language = "python"
