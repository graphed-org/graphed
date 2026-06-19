"""Sphinx configuration for graphed-checkpoint."""

from __future__ import annotations

project = "graphed-checkpoint"
author = "graphed-org"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "furo"
html_title = "graphed-checkpoint"
autodoc_typehints = "description"
autosummary_generate = True
autosummary_imported_members = False
# the dev-extra backends are not installed in the docs job; keep autodoc from importing them
autodoc_mock_imports = ["numpy"]
