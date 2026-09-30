"""Simulate the planned V2 map/gather over the m68c harness data with the planned take() fix."""
import sys; sys.path.insert(0, "tests/frozen/awkward/m68c")
import awkward as ak, numpy as np, graphed
import graphed.awkward.join as J
from graphed.awkward import AwkwardBackend, gak
from graphed.shuffle import JOINKEY
from m68c_services_harness import *
orig = J.take
def fixed(block, index):
    idx = np.asarray(index)
    if len(idx) == 0: return block[:0]
    if len(block) == 0:
        return ak.Array(ak.contents.IndexedOptionArray.simplified(ak.index.Index64(np.full(len(idx), -1, dtype=np.int64)), block.layout))
    return orig(block, index)
J.take = fixed
be = AwkwardBackend()
def maps(chunks, P):
    out = []
    for p in chunks.partitions(2):
        blk = chunks.read_partition(p, None, None)
        k = be.eval_stage("pack_key", [blk], {"on": "run"})
        out.append([be.from_wire(be.to_wire(b)) for b in be.partition(k, JOINKEY, P)])
    return out
E, L = Chunks("events", *events_files()), Chunks("lumi", *lumi_files())
for grouped in (False, True):
  for how in (("inner","left","right","outer") if not grouped else ("inner","left")):
    em, lm = maps(E, 2), maps(L, 2)
    p = {"on": f"run,{JOINKEY}", "how": how, "grouped": grouped}
    outs = []
    for d in range(2):
        l = be.concat([m[d] for m in em]); r = be.concat([m[d] for m in lm])
        jb = be.eval_stage("join", [l, r], p)
        if not grouped: jb = ak.with_field(jb, jb.run * 10, "run10")
        outs.append(pickle_rt := __import__("pickle").loads(__import__("pickle").dumps(jb)))
    ev, lu = two_sources()
    j = gak.join(ev, lu, on=["run"], how=how, grouped=grouped)
    if not grouped: j = gak.with_field(j, j.run * 10, "run10")
    exp = ev.session.materialize(j)
    u = ak.concatenate(outs)
    print(grouped, how, str(u.type) == str(exp.type), (sublists if grouped else rows)(u) == (sublists if grouped else rows)(exp), [len(o) for o in outs], "" if str(u.type)==str(exp.type) else (str(u.type), str(exp.type)))
