"""m58 integ-m58-H3/H4/H5 — where the hook runs, and what it may not reorder.

The declaration is a BUILD-time question: the built plan carries the answer, so no worker ever
consults the source object again, and the same program built twice gives byte-identical plans whose
read list is the source's own order, duplicates included. `read_columns_by_label` has no source
object at all and is untouched by any of it.
"""

from __future__ import annotations

import pickle

from m58_declaration_fixtures import (
    AGGREGATE_VALUE,
    DECLARED,
    VARIED_COLUMNS,
    DeclaringSource,
    PlainSource,
    aggregate_over,
    partitioned,
    varied_record,
)

import graphed
from graphed.core.execution import SequentialRunner


def test_the_built_plan_carries_the_answer_and_no_worker_asks_again() -> None:
    source = DeclaringSource(max_calls=1)  # a second call raises, wherever it comes from
    plan, _outputs = aggregate_over(source)
    assert source.calls == 1

    assert SequentialRunner().run(plan).value == AGGREGATE_VALUE
    assert source.calls == 1  # the run consulted nothing

    shipped = pickle.loads(pickle.dumps(plan))  # the closure a process pool would ship
    assert shipped.process.columns == DECLARED
    assert SequentialRunner().run(shipped).value == AGGREGATE_VALUE
    assert shipped.process.reader.calls == 1  # the count travelled; the hook did not run again
    assert shipped.process.reader.seen == [DECLARED, DECLARED]


def test_two_builds_agree_byte_for_byte_and_keep_the_sources_own_order() -> None:
    first, _outputs = aggregate_over(DeclaringSource())
    source = DeclaringSource()
    second, _outputs = aggregate_over(source)

    assert bytes(first.process.ir) == bytes(second.process.ir)
    assert first.process.columns == second.process.columns == DECLARED
    # graphed neither sorts nor dedupes: "z" leads and "x" arrives twice, exactly as declared
    assert SequentialRunner().run(second).value == AGGREGATE_VALUE
    assert source.seen == [DECLARED, DECLARED]


def test_read_columns_by_label_is_the_same_with_and_without_a_hook() -> None:
    answers = []
    sources = (PlainSource(), DeclaringSource())
    for source in sources:
        _session, root = partitioned(source)
        record, _mask = varied_record(root)
        answers.append(graphed.read_columns_by_label([record], root.node_id))

    expected = dict.fromkeys(("nominal", "murf_1", "murf_5em1"), VARIED_COLUMNS)
    assert answers[0] == answers[1] == expected
    assert sources[1].calls == 0  # a syntactic reader has no source object to ask
