Current limitations
===================

- **Synthetic events only.** The checks run on a deterministic synthetic NanoAOD-like
  dataset, not real CMS Open Data. That fully exercises the "same answer as plain awkward
  on the same input" contract; reading real ROOT files is tested outside this suite.
- **Scale factors are analytic stand-ins.** The b-tag and photon scale factors in the
  systematics fixtures are simple formulas, not real correctionlib JSON or ONNX
  evaluations. ``graphed`` itself supports the real ones; the reference suite just does
  not carry the payload files.
- **Graph-size figures are estimates.** The node counts in :doc:`graph_bloat_note` are an
  operator-count proxy, not measured dask-awkward low-level graph sizes.
