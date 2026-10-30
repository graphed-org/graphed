"""One process of a resume: ``key_order_leg.py <variant> <store> <data dir> <dest>``.

Prints a JSON report; the caller picks the hash seed.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import awkward as ak
import key_order_helpers as h

from graphed import Session, aggregate_plan, repartition, shuffle_plan
from graphed.awkward import from_parquet, gak, to_parquet
from graphed.checkpoint import Store, resumable, run_shuffle_resumable
from graphed.core import SequentialRunner

variant, store, data, dest = sys.argv[1:]
source = os.path.join(data, "events.parquet")
if not os.path.exists(source):
    ak.to_parquet(ak.Array({"x": [float(i) for i in range(40)]}), source)
session = Session(h.backend())
events = from_parquet("events", source, steps_per_file=4, session=session)
fold: dict[str, Any] = {"reduce": h.fold, "combine": h.add, "empty": h.zero}

report: dict[str, Any] = {"order": [repr(key) for key in h.BEHAVIOR]}
if variant == "shuffle":
    staged = shuffle_plan(gak.sum(repartition(events, n=2).x), **fold)
    resumed = run_shuffle_resumable(staged, Store(store))
    report |= {
        "reused": resumed.report.skipped,
        "tasks": sum(len(stage.tasks) for stage in staged.stages),
        "bytes": staged.to_bytes().hex(),
    }
else:
    if variant == "by_value":
        plan = aggregate_plan(gak.sum(events.x), **fold)
    elif variant == "by_ref":
        plan = aggregate_plan(gak.sum(events.x), backend="key_order_helpers:backend", **fold)
    elif variant == "by_factory":
        plan = aggregate_plan(gak.sum(events.x), backend=h.backend, **fold)
    else:
        ref = h.BEHAVIOR if variant == "write_dict" else "key_order_helpers:BEHAVIOR"
        plan = to_parquet(events.x, dest, compute=False, behavior=ref)
    rp: Any = resumable(plan, store)
    SequentialRunner().run(rp)
    report |= {"reused": rp.process.reused, "tasks": len(rp.tasks)}
print(json.dumps(report))
