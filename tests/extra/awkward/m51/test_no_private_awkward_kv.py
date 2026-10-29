"""§6.4e's manifest merge must ride the PUBLIC arrow surface, not awkward's private internals.

The frozen suite pins WHAT lands in the parquet KV (`raw_manifest`, `ak.from_parquet` round-trip);
this guards HOW it gets there — a later "just reach into `awkward._connect`" refactor would still
pass the frozen anchors but couples us to a private module. A source scan (not an import hook) so
it fires on the literal text regardless of lazy-import timing; the public-route control keeps it
non-vacuous.
"""

from __future__ import annotations

import ast
import inspect

import graphed.awkward._varied_read as read_mod
import graphed.awkward._varied_write as write_mod
import graphed.awkward.io as io_mod


def _imported_modules(source: str) -> set[str]:
    mods: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def test_varied_write_imports_no_private_awkward_module() -> None:
    sources = {mod.__name__: inspect.getsource(mod) for mod in (io_mod, read_mod, write_mod)}
    private = {
        (name, m)
        for name, source in sources.items()
        for m in _imported_modules(source)
        if m.startswith("awkward._")
    }
    assert not private, f"varied write reaches into private awkward internals: {sorted(private)}"
    # positive control: the module really does merge the KV through the PUBLIC arrow route, so the
    # scan above ran over real source rather than trivially passing on an empty/unread file.
    assert "replace_schema_metadata" in sources[write_mod.__name__]
