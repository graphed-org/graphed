# Test Dispute — awkward m11 pins the base proxy, which ak.* dispatch (GW6) replaces

**Filed by:** gw6-impl, 2026-10-14, from HiggsDNA-survey finding GW6 (`ak.<fn>(deferred)` must record
`gak.<fn>`).
**Status:** RESOLVED by owner ruling 2026-10-14: "Yes, refreeze." Correction applied and re-frozen as
`freeze-m11-gw6`.

## The tests

`tests/frozen/awkward/m11/test_functions_only_idiom.py`:

| test | assert pinned | correction |
|---|---|---|
| `test_awkward_sessions_return_the_base_neutral_proxy` | `type(a) is Array`, `type(a + a) is Array` | `isinstance(a, Array) and type(a) is AwkwardArray`, `type(a + a) is AwkwardArray` |
| `test_gak_functions_record_over_the_base_proxy` | `type(out) is Array` | `isinstance(out, Array) and type(out) is AwkwardArray` |

## The clause

Design review 2026-06-10 (the module docstring): graphed-awkward supplies no method/property proxy, so
"the awkward backend keeps the base proxy and its user surface stays in the `gak` namespace".

awkward offers third-party arrays an `__awkward_function__` hook, looked up on the array. The owner's
factorization rule keeps idiom surfaces off `graphed.Array`, so the hook lives on the awkward
backend's proxy, `graphed.awkward.AwkwardArray(Array)`, which `AwkwardBackend.array_type` now returns.
The pinned exact type `Array` cannot hold once the backend supplies that proxy.

## The correction (non-weakening)

The proxy is still a `graphed.Array` and still has no member-function idiom; it adds only the hook that
routes `ak.*` to `gak`. The asserts now pin that exact subclass, which is as strict as before: the
base `Array`, or any other subclass, fails them. The docstring sentence that named the base proxy is
reworded to match. `test_no_method_idiom_on_the_proxy_the_surface_is_gak` and every test id are
unchanged.
