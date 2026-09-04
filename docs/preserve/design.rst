How preservation bundles work
=============================

Someone asks you to reproduce a plot from two years ago. Your analysis code has moved on, the
input files were staged off the disk that hosted them, the correction JSON came from a
directory that no longer exists, and the person asking does not have your conda environment.
A preservation bundle is the answer to that request: one directory that carries the analysis,
its inputs and its corrections as data, and hands the histogram back unchanged.

Nothing in it is a new file format. Corrections are correctionlib JSON, models are ONNX or the
framework's own export, histograms follow UHI, and everything is identified by a SHA-256 of its
contents.


Build one and read it back
--------------------------

Save this as ``analysis.py`` — it needs ``pip install "graphed[preserve]"`` and nothing else:

.. code-block:: python

   import json

   import awkward as ak

   from graphed import Session
   from graphed.awkward import AwkwardBackend, from_awkward, gak
   from graphed.preserve import (
       CORRECTIONLIB_PLUGIN,
       build_bundle,
       inspect,
       record_external,
       reproduce,
   )

   # A real correctionlib v2 set: a per-event scale factor binned in jet multiplicity,
   # with one category per systematic.
   SF = {
       "nodetype": "binning",
       "input": "njet",
       "edges": [0.0, 2.0, 4.0, 100.0],
       "content": [0.90, 1.00, 1.10],
       "flow": "clamp",
   }
   CORRECTION = json.dumps(
       {
           "schema_version": 2,
           "corrections": [
               {
                   "name": "event_sf",
                   "version": 1,
                   "inputs": [
                       {"name": "systematic", "type": "string"},
                       {"name": "njet", "type": "real"},
                   ],
                   "output": {"name": "sf", "type": "real"},
                   "data": {
                       "nodetype": "category",
                       "input": "systematic",
                       "content": [{"key": "nominal", "value": SF}],
                   },
               }
           ],
       },
       sort_keys=True,
   ).encode("utf-8")

   events = ak.Array(
       {"Jet": ak.zip({"pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0, 45.0]])})}
   )

   s = Session(AwkwardBackend())
   ev = from_awkward(s, "events", events)
   ht = gak.sum(ev.Jet.pt, axis=1)
   njet = gak.num(ev.Jet, axis=1)
   weight = record_external(
       s, CORRECTIONLIB_PLUGIN, CORRECTION, [njet], params={"name": "event_sf", "systematic": "nominal"}
   )

   HIST = {"name": "ht", "bins": 4, "lo": 0.0, "hi": 200.0}
   payloads = {CORRECTIONLIB_PLUGIN.content_hash(CORRECTION): CORRECTION}
   kwargs = dict(session=s, value=ht, weight=weight, datasets={"events": events},
                 payloads=payloads, histogram=HIST)

   bundle = build_bundle("bundle", **kwargs)
   rebuilt = build_bundle("bundle_again", **kwargs)

   print("same analysis, same fingerprint:", bundle.fingerprint() == rebuilt.fingerprint())
   print("counts:", reproduce(bundle))
   print()
   print(inspect(bundle))

``python analysis.py`` prints::

    same analysis, same fingerprint: True
    counts: [0.  1.9 0.  1.1]

    Preservation Bundle  fingerprint=sha256:caeb3b9fd9234d96dcc0b5817de819edce0bac43144a5e37b8d1ca865547433c
      environment: python 3.13.3; 6 pinned packages
      config: {}   seed: 0
      histogram: {'name': 'ht', 'bins': 4, 'lo': 0.0, 'hi': 200.0}
      graph (IR, opt_level=0):
        n0   source    events         params={} <- []   [analysis.py:52]
        n1   op        field          params={'field': 'Jet'} <- [0]   [analysis.py:53]
        n2   op        field          params={'field': 'pt'} <- [1]   [analysis.py:53]
        n3   op        ak.sum         params={'axis': 1} <- [2]   [analysis.py:53]
        n4   op        ak.num         params={'axis': 1} <- [1]   [analysis.py:54]
        n5   external  external       params={'content_hash': 'sha256:a7b2fc1791b96a22af709e11b4b2d5515f5aef4d696e2449c543aeda4d034b7c', 'framework': 'correctionlib', 'kind': 'correctionlib', 'name': 'event_sf', 'systematic': 'nominal'} <- [4]   [analysis.py:55]
      external payloads (HEP standards, content-addressed):
        n5 correctionlib () sha256:a7b2fc1791b96a22af709e11b4b2d5515f5aef4d696e2449c543aeda4d034b7c
      input datasets:
        events: sha256:cd22b92e153202cc97f5031d4c21cbd4495fd6b4e843d8182955bfd7b523d1b0
      no opaque nodes (every node is durable IR or a content-addressed payload)

Your fingerprint and dataset hash will differ from the ones above — they cover your Python and
package versions and your source filename as well as the analysis — but they are stable across
rebuilds, which is what the first line checks.

Two arguments to ``build_bundle`` are the ones you have to supply, because they cannot be
recovered from a recorded analysis: ``datasets`` maps each source to the array it read, and
``payloads`` maps each correction or model's content hash to its bytes. Everything else — the
graph, the source lines, the environment — comes off the session.

There is a second calling convention. Here the analysis ends in a ``(value, weight, spec)``
triple and the histogram is built at the end; if instead your analysis ends *at* a histogram
fill (the ``graphed-histogram`` path), pass ``value=`` the fill and leave ``weight`` and
``histogram`` out, and ``reproduce`` hands you the histogram itself. Passing one of ``weight``
and ``histogram`` without the other is an error rather than a guess.


What ends up in the directory
-----------------------------

::

    bundle/
      manifest.json     # the bill of materials: names everything, contains nothing heavy
      store/            # every blob, filed under the SHA-256 of its own contents

The manifest lists hashes: the graph's, each dataset's, each correction or model's alongside
its kind and which operation uses it, the source map's, plus the configuration, the seed, and
the environment record. ``Bundle.fingerprint()`` hashes the manifest, giving one identifier for
"exactly this analysis on exactly these inputs".

Filing blobs under a hash of their contents is what makes the bundle checkable rather than
merely tidy. There is no path to go stale, no version string to be wrong, and no way to swap a
correction for a different one without the reference failing to resolve — a substituted or
truncated blob is simply not there under the hash the manifest asks for. It also means the
common case is cheap: two bundles that share a dataset share the bytes.

The graph is stored **unfused**, one entry per operation you wrote. That is not the form a run
wants — it is the form a *reader* wants, and it is why the listing above puts your source line
against every operation. Optimization is the consumer's business: a bundle re-run for real work
reduces the graph first (see below).


Reading an analysis without running it
--------------------------------------

``inspect(bundle)`` produces the listing above from the manifest and source map alone. No
dataset is unpacked, no correction is parsed, no model is loaded. You can run it on a bundle
whose inputs you have no intention of reading, on a machine with none of the ML frameworks
installed, and get a complete account of what the analysis does and what it depends on.

The last line is the one to read first. Most recorded operations are durable data, but a
Python callable you handed in yourself — ``ht.map(lambda a: a)``, say — is not: nothing can
inspect it, hash it meaningfully, or promise it will behave the same in three years. Such an
operation is listed as an **opaque node** and flagged as a preservation risk. It is never
silently dropped and never silently run: you find out at build time, in the listing, not when
the numbers come out different.


Content identity is not byte identity
-------------------------------------

Now that every correction and model is identified by a hash, the obvious question is *a hash of
what?* Not of the file. Re-export an identical model and you get different bytes almost every
time: zip archives stamp timestamps, Keras invents layer names, JAX embeds MLIR source
locations, correctionlib JSON can be pretty-printed or not. Hash the file and you get a bundle
that reports a change every time someone re-saves, and a cache that never hits.

So each payload kind hashes what is actually content. correctionlib hashes the canonical form
of the correction set, not its formatting:

.. code-block:: python

   import json

   from graphed.preserve import CORRECTIONLIB_PLUGIN, sha256_bytes

   CSET = {
       "schema_version": 2,
       "corrections": [
           {
               "name": "event_sf",
               "version": 1,
               "inputs": [{"name": "njet", "type": "real"}],
               "output": {"name": "sf", "type": "real"},
               "data": {
                   "nodetype": "binning",
                   "input": "njet",
                   "edges": [0.0, 2.0, 4.0, 100.0],
                   "content": [0.90, 1.00, 1.10],
                   "flow": "clamp",
               },
           }
       ],
   }

   compact = json.dumps(CSET, sort_keys=True, separators=(",", ":")).encode("utf-8")
   pretty = json.dumps(CSET, indent=4).encode("utf-8")

   edited = json.loads(compact)
   edited["corrections"][0]["data"]["content"] = [0.91, 1.00, 1.10]
   edited_bytes = json.dumps(edited, sort_keys=True, separators=(",", ":")).encode("utf-8")

   print("same bytes?          ", compact == pretty)
   print("same raw sha256?     ", sha256_bytes(compact) == sha256_bytes(pretty))
   print(
       "same content hash?   ",
       CORRECTIONLIB_PLUGIN.content_hash(compact) == CORRECTIONLIB_PLUGIN.content_hash(pretty),
   )
   print(
       "one number changed?  ",
       CORRECTIONLIB_PLUGIN.content_hash(edited_bytes) == CORRECTIONLIB_PLUGIN.content_hash(compact),
   )

Prints::

    same bytes?           False
    same raw sha256?      False
    same content hash?    True
    one number changed?   False

Reformatting does not move the hash; changing one scale factor does. Every shipped payload kind
follows the same rule, going through the framework's own loader to get at the content:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Kind
     - What the hash covers
   * - ``correctionlib``
     - the correction set's canonical contents
   * - ``onnx_model``
     - the weights plus the graph structure
   * - ``tensorflow_model``
     - a ``.keras`` archive's weights plus its config, with generated layer names stripped
   * - ``pytorch_model``
     - a TorchScript archive's sorted ``state_dict`` plus its code
   * - ``xgboost_model``
     - XGBoost's open JSON model, canonicalized
   * - ``jax_export``
     - ``jax.export`` StableHLO with source locations stripped, plus the input signature
   * - ``triton_model``
     - the served model's identity descriptor — see below
   * - ``histogram``
     - the fill's axes and storage

Same weights and same architecture means the same identity, whoever re-exported it and on
whatever day.

``triton_model`` is the exception worth understanding. A Triton-served model lives on someone
else's machine, so what the bundle preserves is the served model's *identity* — name, version,
input and output names, weight digests — while the connection to the server is environment,
built per worker from an importable factory named in the operation's parameters
(``params["transport"] = "module:attr"``; the default builds a ``tritonclient`` HTTP client).
The bundle records what was called. It cannot bottle the server, and if the server is gone,
``reproduce`` says so rather than returning something else.

``sha256_bytes`` is there for the easy case: when the payload bytes already *are* the canonical
content, with no formatting or metadata to normalize away, use it directly.


Adding a format the bundle does not know
----------------------------------------

An ``ExternalPlugin`` is what teaches ``graphed.preserve`` one payload
kind. It answers four questions: how to hash the payload's content, how to load it into a
usable object, how to run that object on inputs, and how to release it. Plus one more —
``samples()``, at least two distinct example payloads — which exists so the hash can be checked
before anyone trusts it.

``register_plugin`` runs that check. It hashes your samples in two subprocesses under different
``PYTHONHASHSEED`` values and refuses a hash that comes out differently, which catches anything
built on ``hash()``, ``id()``, the clock, or randomness — the failure mode where a bundle is
fine on your machine and unresolvable on a colleague's. It also refuses a hash that maps two
distinct samples onto one value, which catches a constant or otherwise useless hash. Both
refusals happen at registration; neither can be discovered later by a wrong number.

``record_external(session, plugin, payload, inputs, params=...)`` then records a call to your
payload in an analysis, exactly the way the example above records a correctionlib call. The
node carries your plugin's content hash, so it is preservable rather than opaque. Build time
and reproduce time both go through ``plugin.evaluate`` on the same bytes, which is why the
bundle comes back bit for bit.

The shipped correctionlib and ONNX plugins are the templates to copy, and
``registered_kinds()`` lists every kind the registry knows, yours included. The frameworks
themselves are imported only when a payload of that kind is actually hashed or evaluated, so a
plugin you never use costs you no dependency.

How a call gets replayed
~~~~~~~~~~~~~~~~~~~~~~~~

Real callees have real signatures. A correction takes a systematic name and several kinematic
arrays; a model takes several tensors, positionally or by keyword; a served endpoint takes
several *named* inputs; a fill takes several axes and possibly several weights. That call shape
is preserved with the operation, as ``params["args"]`` and ``params["kwargs"]``, and replay
obeys it exactly:

* positional — ``"args": [["$0", "$1"], ["$2"]]``, where ``$i`` is graph input *i* and an inner
  list is a group that the ML plugins stack into one ``(n_events, k)`` feature matrix;
* keyword — ``"kwargs": {"mask": ["$2"]}``, genuine Python keyword arguments, for callees with
  Python signatures such as PyTorch and JAX;
* named protocol inputs — ``"args": {"kin": ["$0", "$1"], "mask": ["$2"]}``, for ONNX feeds and
  Triton inputs, where the names belong to the wire protocol rather than to Python;
* constants — correctionlib templates may interleave literals with slots
  (``["nominal", "$0", "$1"]``), which is how a systematic gets selected at any argument
  position. Its array inputs pass through natively: numpy in, numpy out; awkward in, awkward
  out, jagged structure intact.
* histogram fills are described structurally instead — how many axes, whether weighted, how
  many weight inputs — and several weight inputs multiply together into one fill weight.

A plugin whose callee cannot honor a shape says so rather than guessing: XGBoost takes exactly
one tabular matrix, and the named-protocol plugins reject Python keyword arguments. Omitting
the template entirely selects the original single-input convention, so bundles written before
templates existed still mean what they meant.

Loading is done once. A ``ResourceCache`` holds each loaded correction set, model, or
connection for the length of a run and reuses it across every call and every operation that
shares the payload, so a model is not re-read per partition, and everything is closed at the
end.


Every systematic universe from one bundle
-----------------------------------------

Hand ``build_bundle`` a varied value or weight and you get one bundle holding every universe,
not one bundle per universe. ``reproduce`` then returns ``{label: counts}``:

.. code-block:: python

   import tempfile
   from pathlib import Path

   import awkward as ak

   from graphed import Session, labels, vary
   from graphed.awkward import AwkwardBackend, from_awkward, gak
   from graphed.preserve import build_bundle, inspect, reproduce

   events = ak.Array(
       {
           "Jet": ak.zip({"pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0, 45.0]])}),
           "genWeight": ak.Array([1.0, 1.0, 1.0]),
       }
   )

   s = Session(AwkwardBackend())
   ev = from_awkward(s, "events", events)
   ht = gak.sum(ev.Jet.pt, axis=1)
   w = ev.genWeight
   weight = vary(w, "sf", up=w * 1.1, down=w * 0.9)

   HIST = {"name": "ht", "bins": 4, "lo": 0.0, "hi": 200.0}

   with tempfile.TemporaryDirectory() as tmp:
       bundle = build_bundle(
           Path(tmp) / "bundle",
           session=s,
           value=ht,
           weight=weight,
           datasets={"events": events},
           histogram=HIST,
       )
       print("universes recorded:", labels(weight))
       print("universes in the manifest:", sorted(bundle.manifest["analysis"]["variations"]))
       print("listed by inspect without running:", all(x in inspect(bundle) for x in labels(weight)))
       for label, counts in sorted(reproduce(bundle).items()):
           print(f"  {label:<8} {counts}")

Prints::

    universes recorded: ('nominal', 'sf_up', 'sf_down')
    universes in the manifest: ['nominal', 'sf_down', 'sf_up']
    listed by inspect without running: True
      nominal  [0. 2. 0. 1.]
      sf_down  [0.  1.8 0.  0.9]
      sf_up    [0.  2.2 0.  1.1]

Every universe's subgraph is marked as an output, so all of them survive into the bundle, and
the manifest's per-label map is written in sorted order so the manifest bytes — and therefore
the fingerprint — do not depend on the order you declared the variations in. A varied bundle
writes manifest format version 2; an unvaried one keeps version 1 and its single output, so
older bundles read exactly as before. ``inspect`` lists the universes without executing any of
them.


Running it again on new data
----------------------------

``reproduce`` re-runs the preserved analysis on the preserved inputs — that is the
reproducibility guarantee, and it does the plainest possible thing: resolve the graph, bind
each source to its stored dataset, resolve each correction and model through its plugin, and
walk the operations in order. Anything missing raises
:class:`~graphed.preserve.errors.UnresolvedPayload` and the run stops.

Re-*targeting* is the more interesting move, and the bundle supports it because the preserved
graph still carries its output marks. Feed that graph back through the optimizer and it reduces
the way a freshly recorded analysis does — duplicate expressions collapse, unused branches
disappear, and the remaining operations fuse into a handful of stages — and then run those
stages over partitions of *new* input through any executor. A preserved analysis is not a
museum piece; it is a program you can point at this year's dataset.


Not supported yet
-----------------

* **The environment is recorded, not rebuilt.** The manifest pins the Python version and the
  versions of graphed and its array/HEP dependencies; ``reproduce`` runs in whatever
  interpreter you start it in. Record a ``container_digest=`` when you build if you want the
  full environment identified, and run inside that container.
* **Only the unfused graph can be reproduced.** Bundles store one entry per operation you
  wrote. Handing ``reproduce`` an already-fused graph raises rather than guessing.
* **Opaque Python callables are flagged, not preserved.** A ``.map(...)`` over your own
  function is listed as a preservation risk. Record it as a plugin-backed operation instead if
  it needs to survive.
* **Remote models need their server.** A Triton-backed operation reproduces where the endpoint
  exists; there is no way to embed and re-launch the service from the bundle.
* **No export to REANA, CAP, Zenodo or RECAST.** The bundle is the substrate those packagings
  would be built from; nothing writes them today.
* **Behavior classes are not carried.** The reproducing interpreter evaluates through a plain
  backend, so analyses meant for preservation should express, for example, a mass calculation
  as an explicit formula rather than relying on a registered behavior.
* **Datasets are embedded, never referenced.** Every input is copied into the store, which is
  what makes the bundle self-contained and also what makes it as large as its inputs.

See :doc:`improvements` for the limits that are most likely to bite in practice.
