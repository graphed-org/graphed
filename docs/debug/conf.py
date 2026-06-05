"""Sphinx configuration for graphed-debug."""

from __future__ import annotations

project = "graphed-debug"
author = "graphed-org"
release = "0.0.1"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
exclude_patterns = ["_build"]
html_theme = "furo"
html_title = "graphed-debug"
autodoc_typehints = "description"
