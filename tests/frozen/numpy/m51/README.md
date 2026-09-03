# m51 frozen suite — `graphed` / `tests/frozen/numpy/m51`

Milestone m51 (variation-aware write-out). Authority: `systematics-vary-plan.md` r47 (`8cdf7d4`),
§6.4f. This tree holds the ONE m51 numpy anchor: the numpy idiom REFUSES a varied write.

**Awkward-free by construction.** The required 3.14t `test-freethreaded` CI job collects the WHOLE
`tests/frozen/numpy` subtree in one process under `pytest hypothesis numpy` alone (no awkward wheel),
so this file imports neither `awkward` nor `gak`; the `Varied` is built through the numpy idiom
(`graphed.numpy.from_array` + `graphed.vary`). Basename `test_varied_write_refusal.py` is unique
across the whole `tests/frozen/numpy` subtree (checked against the live m5↔m40 `test_projection.py`
duplicate the freethreaded gate tolerates via `--ignore=tests/frozen/numpy/m40`).

| anchor | plan clause | test |
|---|---|---|
| `vary-m51-L` | §6.4f numpy idiom refuses a `Varied` first positional, naming the awkward backend | `test_varied_write_refusal.py::test_numpy_to_parquet_refuses_a_varied_first_positional` |
| `vary-m51-L` | §6.4f entry point is the MODULE path; `graphed.numpy.to_parquet` unexported | `test_varied_write_refusal.py::test_numpy_to_parquet_module_path_is_the_entry_point_not_a_package_attribute` |
| `vary-m51-L` | §6.4f/§10 no `select=` keyword — a `select=` call stays a plain `TypeError` | `test_varied_write_refusal.py::test_numpy_to_parquet_gains_no_select_keyword` |

## Spellings this freeze pins

* The refusal is a **`graphed.errors.GraphedError`** whose message NAMES the awkward backend
  (asserted `match=r"awkward"`); the exact wording is the implementer's, constrained only to name the
  backend it points the analyst at.
* The trigger is a **`Varied` FIRST POSITIONAL** to `graphed.numpy.io.to_parquet` — NOT a `select=`
  kwarg (the numpy idiom gains none; `select=` stays a plain `TypeError`, never a graphed error).

## Non-vacuity

`test_numpy_to_parquet_refuses_a_varied_first_positional` fails against `graphed@main` with an
`AttributeError` (`'session' is not defined on a Varied`): with no guard, `to_parquet` reads
`array.session` first and §2.2's reserved-name rule raises — the exact "guard absent" signal, red
until the awkward-naming guard exists. The other two are positive controls that pass today (guarding
the reverse mistakes: exporting `to_parquet`, or wiring `select=` into the numpy idiom).
