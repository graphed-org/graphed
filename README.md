# graphed

Deferred awkward-array analysis with a plan you can save, ship, and debug — dask-awkward's
shape, without the giant task graph.

You write ordinary awkward code. Nothing runs yet — `graphed` records it, and collapses the
redundancy as you go, so the cut you wrote twice and the helper that multiplies by 1.0 are
gone before they ever cost you anything. When you're ready, you hand the result to a runner.
What you get for it:

- **The same analysis produces byte-identical results every run**, on 1 worker or 100.
- **A failure on a remote worker comes back pointing at the line you wrote**, not an opaque
  string from another process.
- **Only the columns your analysis touches are read off disk** — a file with 400 branches
  costs you the six you used.
- **The recorded plan is a durable artifact**: checkpoint it, resume it after a crash, or
  bundle it so a colleague reproduces your histograms exactly.

## Install

```bash
pip install "graphed[awkward]"      # the HEP default: ragged arrays via awkward
```

Other extras, all opt-in:

```bash
pip install graphed                 # frontend + compiled core only (light)
pip install "graphed[numpy]"        # rectilinear numpy backend
pip install "graphed[parquet]"      # Parquet reading/writing (pyarrow)
pip install "graphed[dashboard]"    # live run dashboard
pip install "graphed[preserve]"     # preservation bundles (correctionlib/ONNX)
pip install "graphed[all]"          # everything except the heavy ML frameworks
```

Wheels ship for Linux, macOS, and Windows; installing from source needs a Rust toolchain
(the optimizer is a compiled extension).

## Your first analysis

```python
import awkward as ak

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak

events = ak.Array(
    [
        {"Jet": [{"pt": 40.0, "eta": 0.1}, {"pt": 12.0, "eta": 2.3}]},
        {"Jet": [{"pt": 55.0, "eta": -1.2}]},
        {"Jet": [{"pt": 8.0, "eta": 0.5}]},
    ]
)

session = Session(AwkwardBackend())
evts = from_awkward(session, "events", events)

good = evts.Jet[evts.Jet.pt > 30.0]          # records, doesn't run
leading_pt = gak.max(good.pt, axis=1)        # gak mirrors ak.* signatures

compiled = graphed.compile_ir(session, leading_pt)
(result,) = graphed.evaluate_ir(compiled, AwkwardBackend(), {"events": events})
print(ak.to_list(result))
# [40.0, 55.0, None]
```

`gak` is the function surface: it carries the `ak.*` signatures you already know
(`gak.num`, `gak.zip`, `gak.with_field`, …), but each call records a step instead of
computing one.

## The one thing that's different

There is no `.compute()`. There are two ways to make the recording happen, and which one you
want depends on where the work should run.

**In this process**, as above: `graphed.compile_ir(session, output)` reduces the recording,
and `graphed.evaluate_ir(...)` evaluates it right here, on arrays you hand it. Good for a
notebook and for small data.

**Across partitions of a dataset**, which is what you want for a real run: build a *plan* and
hand it to a runner. `graphed.aggregate_plan(out1, out2, reduce=..., combine=..., empty=...)`
compiles every output into one recording and produces one task per partition of the dataset
you read from — so it needs a partitioned source such as `from_parquet`. For histograms,
`graphed_histogram.plan({"name": h})` builds the same thing for you. Then:
`SequentialRunner` from `graphed.core` runs it in-process, and a
[`graphed-executors`](https://github.com/graphed-org/graphed-executors) runner runs it on a
process pool, a dask cluster, or a parsl HTEX pool.
[Your first real analysis](docs/quickstart.rst) does exactly this, end to end.

The plan — not a pickle of your Python objects — is what travels, which is why it can be
saved, resumed, and reproduced.

## Which pieces do I need?

| You want | Install / import |
|---|---|
| Ragged HEP analysis (the default) | `graphed[awkward]` → `graphed.awkward`, `gak` |
| Read/write Parquet, skims | add `graphed[parquet]` → `graphed.awkward.from_parquet`, `to_parquet` |
| Run on a pool or cluster | [`graphed-executors`](https://github.com/graphed-org/graphed-executors) |
| Deferred `hist`-style histograms | [`graphed-histogram`](https://github.com/graphed-org/graphed-histogram) |
| Watch a run live | add `graphed[dashboard]` → `graphed.debug` |
| Hand your analysis to a colleague, exactly | add `graphed[preserve]` → `graphed.preserve` |

Everything under one import path, and what each part does for you:

| Import path | What it does for you | Extra |
|---|---|---|
| `graphed` | `Session`, `Array`, `vary` (systematic variations), compile/evaluate | (base) |
| `graphed.core` | the compiled optimizer, the plan types runners consume, and `SequentialRunner` | (base) |
| `graphed.awkward` | ragged backend: `gak` functions, corrections/ONNX calls | `[awkward]` (+ `[parquet]` for I/O) |
| `graphed.numpy` | deferred numpy for rectilinear data | `[numpy]` |
| `graphed.debug` | errors mapped back to your source line; the live dashboard | (base); `[dashboard]` for the live view |
| `graphed.checkpoint` | cache results by content; restart a crashed run where it left off | (base) |
| `graphed.preserve` | a self-contained bundle that reproduces your histograms elsewhere | (base); `[preserve]` for correctionlib/ONNX payloads |

## Next

- [Your first real analysis](docs/quickstart.rst) — a parquet dataset, a jet cut, a
  systematic variation and a histogram, end to end.
- [How the frontend works](docs/frontend/design.rst) — what gets recorded, how duplicate
  expressions collapse as you build, and the full `graphed.vary` grammar for systematics.
- [How the awkward backend works](docs/awkward/design.rst) — column reading, corrections,
  ML models, and writing varied skims.
- [API reference](docs/api.rst).
- Build the docs locally: `pip install -e ".[docs]" && sphinx-build -W -b html docs docs/_build/html`

To hack on `graphed` itself, see [CONTRIBUTING.md](CONTRIBUTING.md).
