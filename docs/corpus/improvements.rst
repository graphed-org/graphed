Improvements
============

Tracked design improvements and known limitations for ``graphed-corpus`` (plan M0 requires this
file in every package).

Current limitations
-------------------

- **Synthetic dataset stand-in.** Fixtures run on a deterministic synthetic NanoAOD-like dataset,
  not real CMS Open Data. This fully exercises ``graphed``'s "same answer as plain awkward"
  contract while staying network-free. See the Required Operations Catalog for the Phase-2
  follow-up (a reduced real AGC/NanoAOD slice + a CMS-published reference).
- **Correction/model stand-ins.** The b-tag and photon scale factors are analytic stand-ins for a
  correctionlib JSON / ONNX model. M3 wraps the real ones as ``External`` nodes carrying a
  content-hashed ``PayloadDescriptor``; M9 preserves them.
- **Graph-bloat numbers are an operator-count proxy.** The exact dask-awkward low-level node counts
  must be measured against the two coffea-benchmark branches (tracked follow-up).

Planned
-------

- ONNX ttbar-reconstruction inference fixture (needs a real model file).
- Systematics-as-a-graph-axis fixtures (named axes / template instantiation).
- A CartesianSelection / >64-category executor stress fixture (PocketCoffea constraint).
