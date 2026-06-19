"""Sphinx configuration for graphed-awkward."""

from __future__ import annotations

project = "graphed-awkward"
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
html_title = "graphed-awkward"
autodoc_typehints = "description"
autodoc_mock_imports = ["correctionlib", "onnx"]
# autosummary recursively generates the API reference (docs/api.rst) from the package, so it never
# drifts from the code. Imported re-exports are documented in their defining module only.
autosummary_generate = True
autosummary_imported_members = False
