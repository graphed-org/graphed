# Test Dispute — `test_jax_hash_is_content_identity`

File: `tests/frozen/preserve/m26/test_ml_plugin_frameworks.py` (graphed#224)

## The test
```python
a, a_reexported = _jax_bytes(0.5), _jax_bytes(0.5)
assert a != a_reexported, "jax.export blobs are volatile — the premise of content hashing"
assert h(a) == h(a_reexported)
```

## The contradiction (measured)
The first assertion is the control that makes `h(a) == h(a_reexported)` non-trivial. jax 0.12.1 (released 2026-11-18) serializes a plain re-export to the same bytes, so the control fails on every PR:

```
jax 0.12.0.1  re-export equal: False  len 2632  2 bytes differ
jax 0.12.1    re-export equal: True   len 2576
```

The 0.12.0.1 difference is the caller frame's `#loc` line; 0.12.1 records no caller frame. The plugin's content hash is not shown wrong.

## Correction
Keep the control, built from a difference both versions record: `_jax_bytes_moved` defines the same function on other source lines. Its bytes differ from `_jax_bytes`'s on 0.12.0.1 and on 0.12.1, and its hash is equal. So `a != a_moved` holds on both, and `h(a) == h(a_reexported) == h(a_moved)` replaces `h(a) == h(a_reexported)`. The weight and structure assertions are unchanged.

## Ruling (owner, 2026-11-19)
"#224 the re-freeze is approved". Amendment under `--allow-refreeze tests/frozen/preserve/m26`, re-tagged `freeze-m26-fixup3` on the merge commit.
