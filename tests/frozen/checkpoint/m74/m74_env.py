"""T3's runs, one per interpreter: ``python m74_env.py <runner> <store> [--accept] [--salt S] [--kill K]
[--error FILE]``; the last stdout line is a JSON report."""

from __future__ import annotations

import argparse
import json
import pickle
from typing import Any

import m74_helpers as h

from graphed.checkpoint import EnvironmentChanged, Store, resumable, run_resumable, run_shuffle_resumable
from graphed.checkpoint.runner import _SimulatedInterrupt
from graphed.core import SequentialRunner

parser = argparse.ArgumentParser()
parser.add_argument("runner", choices=["resumable", "run_resumable", "run_shuffle_resumable"])
parser.add_argument("store")
parser.add_argument("--accept", action="store_true")
parser.add_argument("--salt", default="")
parser.add_argument("--kill", type=int, default=None)
parser.add_argument("--error", default=None)
args = parser.parse_args()

out: dict[str, Any] = {}
h.reset()
try:
    if args.runner == "resumable":
        rp = resumable(h.plan_of(h.marked), args.store, salt=args.salt, accept_environment=args.accept)
        out["reused"] = rp.process.reused
        out["value"] = SequentialRunner().run(rp).value
    elif args.runner == "run_resumable":
        res = run_resumable(
            h.durable("m74_helpers:marked"), Store(args.store), accept_environment=args.accept, _kill_after=args.kill
        )
        out["reused"], out["value"] = res.report.skipped, res.value
    else:
        shuffled = run_shuffle_resumable(
            h.staged(), Store(args.store), accept_environment=args.accept, _kill_after=args.kill
        )
        out["reused"], out["value"] = shuffled.report.skipped, list(shuffled.value)
    out["outcome"] = "ok"
except EnvironmentChanged as exc:
    out.update(outcome="EnvironmentChanged", message=str(exc))
    if args.error:
        with open(args.error, "wb") as f:
            pickle.dump(exc, f)
except _SimulatedInterrupt:
    out["outcome"] = "interrupted"
out["calls"] = h.STATE["calls"]
print(json.dumps(out))
