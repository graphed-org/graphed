Current limitations
===================

What a preservation bundle does not do for you yet, and what to do instead. The full picture of
what it *does* is in :doc:`design`.

The environment is recorded, not recreated
------------------------------------------

The manifest pins your Python version and the versions of graphed and its array and HEP
dependencies — enough to tell a later reader what the numbers were produced with, and enough
that a version change shows up in the fingerprint. It is not a lockfile for your whole
environment, and ``reproduce`` makes no attempt to install anything: it runs in the interpreter
you start it in.

**Workaround:** pass ``container_digest=`` to ``build_bundle`` and reproduce inside that image.
The digest is recorded in the manifest and is part of the fingerprint.

Multi-universe bundles need the value/weight/spec form
------------------------------------------------------

One bundle holds every systematic universe when you pass a varied ``value`` or ``weight``
together with a ``histogram`` spec — that is the form shown in :doc:`design`. The other calling
convention, where the analysis ends *at* a histogram fill and you pass only ``value=``, takes a
single array; handing it a varied one raises rather than silently picking a universe.

**Workaround:** either use the ``value``/``weight``/``histogram`` form, or pick the universe
you want first with ``graphed.universe(v, label)`` or ``graphed.nominal(v)``.

Datasets are embedded, never referenced
---------------------------------------

Every input array is copied into the bundle's store. That is what makes the directory
self-contained and portable, and it also makes the directory as large as the data it read.
There is no "reference the file at this URL and check its hash" mode.

**Workaround:** bundle a slim skim rather than the full input, and preserve the skimming step
as its own bundle.

Remote models reproduce only where the server does
--------------------------------------------------

A Triton-backed operation preserves the served model's identity, not the service. Reproduction
connects to the endpoint named in the operation's parameters; if the server is gone,
``reproduce`` raises. Nothing packages a servable model into the bundle and launches it.

**Workaround:** for a model that has to outlive its serving infrastructure, export it to ONNX
or TorchScript and record it as a local payload.

Only the unfused graph can be reproduced
----------------------------------------

Bundles store the graph one entry per operation you wrote, which is what makes ``inspect``
readable and what puts your source line against every operation. The reproducing interpreter
therefore refuses a graph that has already been fused into stages, rather than guessing at it.
This costs nothing on the reproduce path — but if you want the optimized run, reduce the
preserved graph first and hand the result to an executor, as :doc:`design` describes.

Your own Python callables are flagged, not preserved
----------------------------------------------------

An operation carrying a function of yours — ``ht.map(lambda a: a)`` and anything like it — has
no content anyone can hash and no guarantee of behaving the same later. ``build_bundle``
records it as an opaque node and ``inspect`` reports it as a preservation risk. It is not run
during reproduction and it is not quietly dropped.

**Workaround:** wrap the operation in an
``ExternalPlugin`` and record it with ``record_external``. The
correctionlib and ONNX plugins are the templates; :doc:`design` walks through what a plugin has
to answer.

Installing PyTorch and XGBoost together
---------------------------------------

Both vendor their own OpenMP runtime, and on macOS the two cannot coexist in one process —
importing both aborts with ``OMP: Error #15``. This is a conflict between the two frameworks,
not something the plugins can arbitrate.

**Workaround:** set ``KMP_DUPLICATE_LIB_OK=TRUE`` and ``OMP_NUM_THREADS=1`` before either
import, or keep the two in separate processes.

TorchScript is the PyTorch payload format
-----------------------------------------

``pytorch_model`` payloads are TorchScript archives, which is today's self-contained,
class-free deployment artifact with the widest installed base — but ``torch.jit`` is on its way
out in favour of ``torch.export``, and recent torch versions warn about it. A ``.pt2``
(``torch.export``) payload kind is the planned replacement.

Not built
---------

* HS3 capture for a statistical model, and writing the reproduced histogram back out as UHI.
* Packaging a bundle for REANA, CAP, Zenodo or RECAST. The bundle is the substrate those would
  be built from.
* Carrying a behavior class by reference, so a preserved analysis could call ``.mass`` rather
  than spelling the formula out.
