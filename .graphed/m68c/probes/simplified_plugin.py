import numpy as np, awkward as ak
import graphed.awkward.join as J
orig = J.take
def fixed(block, index):
    idx = np.asarray(index)
    pass
    if len(block) == 0:
        return ak.Array(ak.contents.IndexedOptionArray.simplified(ak.index.Index64(np.full(len(idx), -1, dtype=np.int64)), block.layout))
    return orig(block, index)
J.take = fixed
