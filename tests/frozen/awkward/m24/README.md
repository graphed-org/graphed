# Frozen acceptance suite — M24 (graphed-awkward): gak <-> ak interface parity

User-directed (2026-06-11): every gak function mirroring an awkward op surfaces the SAME
interface for genuinely missing functionality. Traceability:

| Test file | Verifies | Item |
|---|---|---|
| `test_interface_parity.py` | the ANTI-DRIFT pin (shared parameter defaults equal awkward's, by inspection, for 38 functions); default-axis fixes behave (concatenate=0, sort/argsort/local_index=-1, argmin/argmax/count=None); zip sequence→tuple records, depth_limit (the events-of-collections enabler), inline with_name/parameters; combinations/argcombinations replacement+axis(+naming); cartesian nested-list+with_name; reducer keepdims/mask_identity/initial; WEIGHTED mean/var/std/moment vs eager; sort/argsort stable=; drop_none axis=; nan_to_num nan/posinf/neginf; isclose equal_nan; broadcast_arrays depth_limit; zeros/ones_like preserve dtype by default; IR byte-determinism with the new params | M24 |

Deliberately absent everywhere (NOT gaps): `highlevel`/`attrs` (eager concerns) and `behavior`
(backend-owned, M18). Documented Phase 2: zip right_broadcast/optiontype_outside_record,
broadcast_arrays rules, concatenate/where mergebool, *_like/values_astype including_unknown,
unzip how, nan_to_num copy, named-axis kwargs.
