#!/usr/bin/env bash
# Run every package's frozen (+extra) suite, one pytest process per subtree.
#
# Per-subtree isolation is REQUIRED, not cosmetic: a single combined `pytest tests/` cannot collect
# this tree (duplicate test basenames across packages, e.g. test_topologies.py x5) and would break
# frozen guards that assert process-global state (core's `assert "awkward" not in sys.modules`). Each
# subtree reproduces its origin repo's isolated environment, so every suite passes as it did there.
#
#   ./scripts/run-tests.sh          # just run the suites
#   COV=1 ./scripts/run-tests.sh    # + accumulate combined branch coverage over `graphed`, gate >=90
set -uo pipefail
cd "$(dirname "$0")/.."
COV=${COV:-0}

# name : space-separated pytest paths (frozen + any extra) for that package
SUITES=(
  "core:tests/frozen/core"
  "frontend:tests/frozen/frontend tests/extra/frontend"
  "numpy:tests/frozen/numpy tests/extra/numpy"
  "awkward:tests/frozen/awkward"
  "debug:tests/frozen/debug"
  "checkpoint:tests/frozen/checkpoint tests/extra/checkpoint"
  "preserve:tests/frozen/preserve"
  "corpus:tests/frozen/corpus"
)

# Packages that must run one process PER MILESTONE subdir (not one per package): they aggregate
# origin-repo milestones with duplicate top-level module basenames (M40's shuffle_backends.py /
# test_projection.py), which collide when collected in a single prepend-import process.
SPLIT_PKGS="frontend numpy awkward"

rc=0
[ "$COV" = 1 ] && rm -f .coverage .coverage.*

run_one() {  # run ONE pytest process over the given paths; accumulate rc + combined coverage
  echo "═══════════════ $1 ═══════════════"; shift
  if [ "$COV" = 1 ]; then
    # --cov-fail-under=0: no per-subtree gate (each subtree alone is far under 90); the gate is the
    # combined `coverage report --fail-under=90` after all subtrees have appended.
    python -m pytest "$@" -q --cov=graphed --cov-append --cov-branch --cov-fail-under=0 || rc=1
  else
    python -m pytest "$@" -q || rc=1
  fi
}

for entry in "${SUITES[@]}"; do
  name=${entry%%:*}; paths=${entry#*:}
  # keep only paths that exist (extra/ dirs are not present for every package)
  existing=""; for p in $paths; do [ -e "$p" ] && existing="$existing $p"; done
  [ -z "$existing" ] && continue
  case " $SPLIT_PKGS " in
  *" $name "*)
    # Finer (per-milestone) isolation: these packages aggregate origin-repo milestones with DUPLICATE
    # top-level basenames — M40 added m39/m40 `shuffle_backends.py` (frontend) and m5/m40
    # `test_projection.py` (numpy, awkward). Collected in ONE process, prepend-import caches the first
    # under its bare name and shadows/mismatches the second -> ImportError / "import file mismatch". So
    # run each frozen/<pkg>/<milestone> in its OWN process; cross-dir helper imports still resolve via
    # the pytest `pythonpath`. extra/<pkg> runs as its own process too.
    for d in tests/frozen/$name/*/; do run_one "$name/$(basename "$d")" "${d%/}"; done
    [ -e "tests/extra/$name" ] && run_one "$name/extra" "tests/extra/$name"
    ;;
  *)
    run_one "$name" $existing
    ;;
  esac
done

if [ "$COV" = 1 ]; then
  echo "═══════════════ combined coverage ═══════════════"
  coverage report --fail-under=90 || rc=1
fi
exit $rc
