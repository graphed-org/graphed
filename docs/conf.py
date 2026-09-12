"""Sphinx configuration for the consolidated graphed package.

One project covering every subpackage (graphed.core, .awkward, .numpy, .debug, .checkpoint,
.preserve) plus the M0.5 requirements corpus (docs-only, no shipped graphed.corpus module —
see MIGRATION.md).
"""

from __future__ import annotations

import shutil

project = "graphed"
author = "graphed-org"
release = "0.0.2"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]

# inheritance_diagram shells out to the `dot` binary; only enable it when Graphviz is on PATH.
if shutil.which("dot"):
    extensions.append("sphinx.ext.inheritance_diagram")

templates_path = ["_templates"]
# docs/corpus/requirements/ holds the internal operations catalog the reference-suite check
# reads; it is not written for readers and is not part of the rendered documentation.
exclude_patterns = ["_build", "corpus/requirements"]

html_theme = "furo"
html_title = "graphed"

# Notebook outputs are committed; the docs build has no executors or ROOT I/O to re-run them with.
nb_execution_mode = "off"

# myst-nb parses both Markdown pages and notebooks; its filetype is "myst-nb", not "markdown".
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

autodoc_typehints = "description"
# autosummary recursively generates the API reference (docs/api.rst) from the package itself, so
# it can never drift from the code. Imported re-exports are documented in their defining module
# only (e.g. GraphStore is documented under graphed.core.graphed_core, not re-listed under
# graphed.core).
autosummary_generate = True
autosummary_imported_members = False

# Optional/heavy deps that may not be installed wherever docs are built; numpy and awkward are
# hard runtime requirements of this package and are left unmocked.
autodoc_mock_imports = [
    "torch",
    "tensorflow",
    "jax",
    "xgboost",
    "tritonclient",
    "keras",
    "perspective",
    "tornado",
    "websocket",
    "correctionlib",
    "onnx",
    "onnxruntime",
    "boost_histogram",
    "hist",
    "vector",
    "pandas",
    "cloudpickle",
    "pyarrow",
]
