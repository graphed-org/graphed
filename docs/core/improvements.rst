Current limits
==============

What ``graphed.core`` does not do yet, and what to do instead.

The optimizer will not rewrite your algebra
-------------------------------------------

The rewrite rules are limited to two things it can prove: argument order does not matter for
``add`` / ``mul`` / ``and`` / ``or`` / ``eq`` / ``ne`` / ``maximum`` / ``minimum``, and adding
``0.0`` or multiplying by ``1.0`` is a no-op. Regrouping a long sum, folding constants together,
and anything that depends on array semantics — mask fusion, field collapse — are not attempted,
because a rule that is subtly wrong corrupts every analysis that trips it.

The practical effect is small: those two rules cover what actually accumulates in analysis code
(a cut written twice, a helper that scales by a weight of 1.0). If you have an expression you
know is cheaper in another form, write it in that form.

Fusion is structural, not cost-based
------------------------------------

Stage boundaries are decided from the shape of the graph, not from a model of what each
operation costs in kernel time or memory. Your only lever is ``maximal_fusion=True``, which
additionally fuses a value whose consumers all land in one stage — trading one recomputation for
one fewer dispatch. There is no per-stage cost model choosing between them for you.

Recording from many threads serializes at one lock
--------------------------------------------------

One lock guards the table that gives structurally identical expressions the same node. Under
free-threaded CPython the critical section is a hash lookup and a push, and stress testing has
not shown it to matter, but a workload that genuinely contends on *recording* — many threads
building unrelated graphs into one store at once — will queue there. Building into separate
stores avoids it entirely; the reduced graphs are byte-identical either way.

One durable graph format version
--------------------------------

Saved graphs carry a version tag and only the current one is accepted. A blob carrying anything
else is rejected with an error rather than misread, and the tag is the hook that will let a
future format be added without breaking what you have already written. Plans and content hashes
you write today stay readable.
