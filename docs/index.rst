graphed
=======

A schedulable, serializable, debuggable HEP task-graph system. ``graphed`` records ordinary
array expressions into an interned Rust IR, reduces the graph to a concise set of fused stages via
equality saturation **as the user builds it** — so a large un-reduced graph never has to exist — and
then executes, checkpoints, debugs, and preserves the result.

One distribution now spans the whole pipeline:

- a deferred-array **frontend** with two backends — ``graphed.awkward`` (ragged data, the HEP
  default, with ``ak.*``-parity ``gak`` operations) and ``graphed.numpy`` (rectilinear), each
  exposing a broad, dask/dask-awkward-level API surface;
- a Rust+PyO3 **optimiser** (hash-consing, DCE/CSE, equality-saturation stage fusion) whose
  reduction is incremental, deterministic, and CI-guarded against super-linear scaling;
- source-mapped **debugging** — a remote-worker failure re-raised at the user's analysis line — plus
  a live dashboard;
- content-addressed **checkpoint/resume** and a self-contained **preservation bundle** that
  reproduces histograms bit-for-bit on a clean machine, with ML-framework
  (``torch``/``tf``/``xgboost``/``jax``/ONNX/Triton) ``External`` plugins.

The distribution is a single pip package, ``graphed``, consolidated from what were formerly separate
prototype repositories (``graphed_core`` → ``graphed.core``, and likewise for
``awkward``/``numpy``/``debug``/``checkpoint``/``preserve``). Start with :doc:`architecture` for the
repository structure and the map of subpackages; each subpackage below then keeps its own design
writeup, and :doc:`api` is the generated reference for the whole distribution.

Old import path → new import path
---------------------------------

===============================================  ========================  =================
Old package (dist / import)                      New import path           Install extra
===============================================  ========================  =================
``graphed`` (frontend)                           ``graphed``               (base)
``graphed-core`` / ``graphed_core``              ``graphed.core``          (base)
``graphed-awkward`` / ``graphed_awkward``        ``graphed.awkward``       ``[awkward]``
``graphed-numpy`` / ``graphed_numpy``            ``graphed.numpy``         ``[numpy]``
``graphed-debug`` / ``graphed_debug``            ``graphed.debug``         ``[dashboard]``
``graphed-checkpoint`` / ``graphed_checkpoint``  ``graphed.checkpoint``    ``[checkpoint]``
``graphed-preserve`` / ``graphed_preserve``      ``graphed.preserve``      ``[preserve]``
===============================================  ========================  =================

.. toctree::
   :maxdepth: 1
   :caption: Overview

   architecture

.. toctree::
   :maxdepth: 1
   :caption: Subpackages

   frontend/index
   core/index
   awkward/index
   numpy/index
   debug/index
   checkpoint/index
   preserve/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api
   corpus/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
