Improvements
============

Tracked design improvements and limitations for ``graphed.numpy`` (plan M0 requires this file).

Current limitations
-------------------

- **Trivial seam-prover.** Operates on 1-D numpy bags only; it exists to prove the backend
  boundary, not to be a production backend (that is ``graphed.awkward``, M3).
- **Opaque ``map`` payloads are not content-addressed.** ``external_payload`` flags wrapped
  callables as a preservation risk; real content hashing is M9.
- **op_form over-declares join nullability via ``__valid_<n>__`` companions (sound, M40).** ``NumpyForm``
  has no option type, so a non-inner ``join`` op_form records that an exclusive field became nullable by
  appending a boolean ``__valid_<n>__`` companion column; the materialized ``MaskedArray`` carries the
  same nullability in ``.mask`` (no companions). op_form's field set is thus a *superset* of the block's
  — a sound over-declaration, never an under-declaration (§A.3.1). Dropping the companions would
  under-declare; materializing them into every block would duplicate ``.mask`` for a form-only artifact
  no consumer reads. A form-vs-block check here must assert ``op_form.fields`` ⊇ ``block.dtype.names``,
  not equality.
- **A null join key is unrepresentable in the packed key (M40).** ``pack_key``/``_as_columns`` drop the
  mask, so a null key concretizes (``event=None`` → ``event=0``) and would collide with a genuine
  ``event=0`` row — a narrow durable-form gap confined to this trivial seam-proving backend, unreachable
  via the pinned ``(run, lumi, event)`` ``__joinkey__`` flow (never null). The minimal sentinel-safe
  hardening, if this backend is ever taken to real use, is to reject or propagate a masked key rather
  than concretize it; bundle it with any future numpy-backend hardening, not its own round.

Planned
-------

- Column projection via ``Backend.project`` (M5) and participation in the execution contract (M7).
