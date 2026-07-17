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

rc=0
[ "$COV" = 1 ] && rm -f .coverage .coverage.*
for entry in "${SUITES[@]}"; do
  name=${entry%%:*}; paths=${entry#*:}
  # keep only paths that exist (extra/ dirs are not present for every package)
  existing=""; for p in $paths; do [ -e "$p" ] && existing="$existing $p"; done
  [ -z "$existing" ] && continue
  echo "═══════════════ $name ═══════════════"
  if [ "$COV" = 1 ]; then
    # --cov-fail-under=0: no per-subtree gate (each subtree alone is far under 90); the gate is the
    # combined `coverage report --fail-under=90` after all subtrees have appended.
    python -m pytest $existing -q --cov=graphed --cov-append --cov-branch --cov-fail-under=0 || rc=1
  else
    python -m pytest $existing -q || rc=1
  fi
done

if [ "$COV" = 1 ]; then
  echo "═══════════════ combined coverage ═══════════════"
  coverage report --fail-under=90 || rc=1
fi
exit $rc
