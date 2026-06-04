"""Sphinx configuration for graphed."""

from __future__ import annotations

project = "graphed"
author = "graphed-org"
release = "0.0.1"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
exclude_patterns = ["_build"]
html_theme = "furo"
html_title = "graphed"
autodoc_typehints = "description"
