"""Sphinx configuration for graphed-awkward."""

from __future__ import annotations

project = "graphed-awkward"
author = "graphed-org"
release = "0.0.1"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
exclude_patterns = ["_build"]
html_theme = "furo"
html_title = "graphed-awkward"
autodoc_typehints = "description"
autodoc_mock_imports = ["correctionlib", "onnx"]
