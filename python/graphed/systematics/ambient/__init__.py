"""The ambient event weight (§2.1-§2.3): the operation list, its provenance, its composition.

An event context's `AmbientRecord` holds this state — the ordered operations, the slot ids, the
riders, the reads, the memo. This package alone builds and mutates it, and holds every rule that
reads it: which of the three outcomes a weight registration decides (a read it names, a factor it
joins, a new factor), how an entry is re-indexed across a row-space change, and how the operations
compose into one container. Moved out of `graphed.context` (m57): the context tree is one concern,
what rides it is another.

Every decision here reads PROVENANCE — the rider beside a slot, the recorded reads — and never the
entry object as it currently stands, which a re-index replaces with a bare member.

- `record` — the record types, and how a record starts.
- `entries` — the live operation list, each entry's rider, its re-index across row-space links.
- `materialise` — reading the ambient: the composed container, its memo, the recorded reads.
- `extension` — what a weight registration's central names, and its refusals.
- `join` — the join and overlay outcomes.
- `forms` — `vary`'s weight and shift forms.
- `compose` — §4's composition walk, over arrays and over forms.
"""

from __future__ import annotations

from .compose import two_level as two_level
from .entries import (
    ambient_entries as ambient_entries,
)
from .entries import (
    live_factors as live_factors,
)
from .entries import (
    live_riders as live_riders,
)
from .entries import (
    row_links as row_links,
)
from .forms import shift_form as shift_form
from .forms import weight_form as weight_form
from .materialise import adopt as adopt
from .materialise import fold as fold
from .materialise import weight_of as weight_of
from .record import (
    AmbientRecord as AmbientRecord,
)
from .record import (
    Registration as Registration,
)
from .record import (
    Rider as Rider,
)
from .record import (
    context_labels as context_labels,
)
from .record import (
    seed as seed,
)
