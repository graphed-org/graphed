Current limitations
===================

- **Flat arrays only.** Data is columns of fixed-width numbers with one partitioned event
  axis (fixed trailing axes are fine). For ragged data — jets per event, muons per event —
  use :doc:`graphed.awkward <../awkward/index>`; it shares the same session and runners.
- **A callable you wrap yourself is identified by name, not content.** ``apply_gufunc`` and
  ``.map`` record the ``name=`` you give them; your function body never enters the graph. Two
  different functions recorded under one name therefore intern to the same node, and the second
  call silently gets the first one's result. A preservation bundle flags such a node as opaque
  rather than reproducing it. Give every wrapped callable its own name, and keep
  reproducibility-critical steps in the recorded ``np.*`` surface where you can.
