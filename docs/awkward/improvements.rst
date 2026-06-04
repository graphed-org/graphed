Improvements
============

Tracked design improvements and limitations for ``graphed-awkward`` (plan M0 requires this file).

Current limitations
-------------------

- **External op output forms are approximated** by the first input's form (corrections/inference are
  ~shape-preserving for the corpus fixtures); precise output-form inference for ONNX is future work.
- **Reference evaluation only.** ``materialize`` runs node-by-node; the morsel-driven executor is M7.
  Column projection via the reporting typetracer is M5.
- **No real ONNX/correctionlib evaluation in tests.** The descriptors are content-hashed and the
  External nodes recorded; running the actual model/correction is exercised end-to-end in M7.

Planned
-------

- Column projection (M5) using the reporting typetracer to collect touched form-keys.
- A real ``from_root`` (uproot metadata) source with a dataset descriptor.
