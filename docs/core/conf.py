"""Sphinx configuration for graphed-core."""

from __future__ import annotations

project = "graphed-core"
author = "graphed-org"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "graphed-core"

autodoc_typehints = "description"
# The API is a small flat set of Rust-backed classes (no Python inheritance hierarchy), so no
# inheritance diagram is meaningful here.
