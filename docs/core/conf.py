"""Sphinx configuration for graphed-core."""

from __future__ import annotations

project = "graphed-core"
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
html_title = "graphed-core"

autodoc_typehints = "description"
# autosummary recursively generates the API reference (docs/api.rst) from the package itself —
# including the Rust-backed graphed_core.graphed_core submodule — so it never drifts from the code.
autosummary_generate = True
autosummary_imported_members = False
# The API is a small flat set of Rust-backed classes (no Python inheritance hierarchy), so no
# inheritance diagram is meaningful here.
