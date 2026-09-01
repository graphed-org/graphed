"""§2.5: a registered label reaching no marked output is REPORTED, not raised.

DCE already prunes the work; the diagnostic is what stops the mkShapesRDF silent-cost case, where a
systematic is registered, paid for in build time, and quietly never filled. The channel is an
additive `CompiledGraph` field — §7.2's schema-absence anchor is worded over `ExecResult`/`Plan`/the
monitor payload, not over `CompiledGraph`.
"""

from __future__ import annotations

from vary_fixtures import loose_varied, sibling_outputs, vector_source

import graphed
from graphed import compile_ir


def test_a_registered_label_that_reaches_no_marked_output_is_reported() -> None:
    session, x = vector_source()
    filled = loose_varied(x, "jes")
    dropped = loose_varied(x, "jer")  # registered, then never marked as an output
    compiled = compile_ir(session, *sibling_outputs(filled))
    unreached = compiled.unreached_labels
    assert set(unreached) >= set(graphed.labels(dropped)) - {"nominal"}
    assert not set(unreached) & (set(graphed.labels(filled)) - {"nominal"})
    assert list(unreached) == sorted(unreached)  # sorted, so the report is order-deterministic


def test_no_label_is_reported_when_every_label_reaches_an_output() -> None:
    session, x = vector_source()
    compiled = compile_ir(session, *sibling_outputs(loose_varied(x, "jes")))
    assert compiled.unreached_labels == ()


def test_the_report_is_a_diagnostic_and_the_compile_still_succeeds() -> None:
    """A raise here would break a legal program: a label whose universe the user chose not to fill
    is valid, only expensive."""
    session, x = vector_source()
    loose_varied(x, "jer")
    compiled = compile_ir(session, x * 2.0)
    assert compiled.ir  # the compile produced a real artifact
    assert compiled.unreached_labels
