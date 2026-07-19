Improvements
============

Tracked design improvements and limitations for ``graphed.awkward`` (plan M0 requires this file).

Current limitations
-------------------

- **External op output forms are approximated** by the first input's form (corrections/inference are
  ~shape-preserving for the corpus fixtures); precise output-form inference for ONNX is future work.
- **Reference evaluation only.** ``materialize`` runs node-by-node; the morsel-driven executor is M7.
  Column projection via the reporting typetracer is M5.
- **No real ONNX/correctionlib evaluation in tests.** The descriptors are content-hashed and the
  External nodes recorded; running the actual model/correction is exercised end-to-end in M7.
- **op_form over-declares an option join key (sound, M40).** ``join_form`` marks the coalesced ``on``
  key option-typed iff *either* input key is option, for every ``how``. That is a supertype of the
  tight per-``how`` type (``inner`` never null; ``left``/``right`` follow that side; ``outer`` either),
  so op_form can never *under*-declare (§A.3.1); ``merge_records`` refines to the exact non-option
  runtime type when no null survives. Kept uniform deliberately — the per-``how`` rule is complexity for
  a durable type already a correct supertype and re-derived exactly at runtime.
- **Both-null join keys match, like ``pandas.merge`` (M40).** A join whose key is null on *both* sides
  pairs the null rows — matching ``pandas.merge`` on ``NaN``/``pd.NA`` keys (the relational reference),
  diverging only from strict SQL ``NULL != NULL``. Declined deliberately: matching the pandas oracle is
  the intended semantics, the awkward null fill (``INT64_MAX``) exceeds the packable key range so a null
  never collides with a *real* key, and a real HEP key ``(run, lumi, event)`` is never null. Do not add a
  frozen test pinning both-null → 1 row — it would pin against the pandas reference.

Planned
-------

- Column projection (M5) using the reporting typetracer to collect touched form-keys.
- A real ``from_root`` (uproot metadata) source with a dataset descriptor.
