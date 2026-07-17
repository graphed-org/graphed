"""Sphinx configuration for graphed-corpus."""

from __future__ import annotations

project = "graphed-corpus"
author = "graphed-org"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "graphed-corpus"

autodoc_typehints = "description"
autosummary_generate = True
autosummary_imported_members = False
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
