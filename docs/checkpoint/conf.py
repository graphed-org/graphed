"""Sphinx configuration for graphed-checkpoint."""

from __future__ import annotations

project = "graphed-checkpoint"
author = "graphed-org"
release = "0.0.1"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
exclude_patterns = ["_build"]
html_theme = "furo"
html_title = "graphed-checkpoint"
autodoc_typehints = "description"
# the dev-extra backends are not installed in the docs job; keep autodoc from importing them
autodoc_mock_imports = ["numpy"]
