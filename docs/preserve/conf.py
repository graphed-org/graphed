"""Sphinx configuration for graphed-preserve."""

from __future__ import annotations

project = "graphed-preserve"
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
html_title = "graphed-preserve"
autodoc_typehints = "description"
autosummary_generate = True
autosummary_imported_members = False
# heavy / lazily-imported runtime libs need not be importable to render the API
autodoc_mock_imports = [
    "awkward",
    "numpy",
    "correctionlib",
    "onnx",
    "onnxruntime",
    "graphed_awkward",
    "graphed_corpus",
    "tensorflow",
    "torch",
    "xgboost",
    "jax",
    "tritonclient",
]
