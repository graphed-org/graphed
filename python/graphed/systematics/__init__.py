"""`graphed.systematics`: the variation machinery — what `graphed.vary` registers and what rides.

The verbs stay where analyses already read them: `graphed.vary`, `graphed.weight`,
`graphed.variations`, `graphed.explain` and the rest of §9.1 are exported from `graphed` itself, and
`graphed.varied`/`graphed.accessors`/`graphed.vary` keep working as module paths. This package is
where the machinery LIVES, one concern per module:

- `registration` — `graphed.vary`'s three overloads: the checks, the minting, the fan-out.
- `ambient` — the ambient event weight: the operation list, each entry's rider, the three outcomes
  a weight registration decides between, and the composition itself.
- `explain` — `graphed.explain`'s record of what varies here and how the families relate.
- `varied` — the `Varied` container, its surface dispositions and its re-wrapping rules.
- `accessors` — §9.1's reader verbs (`labels`, `universe`, `nominal`, `weight`, `variations`, …).
- `by_label` — per-label column reads and impact.
- `kinds`, `tags`, `points` — a variation's kind, a universe's tag spelling, a family's points.

The event context itself is NOT here: it is a frontend object with a row space and collections
(`graphed.context`), and it KEEPS the ambient state that `ambient` decides over.
"""

from __future__ import annotations

from .accessors import (  # isort: skip
    broadcast_like as broadcast_like,
    context_of as context_of,
    labels as labels,
    nominal as nominal,
    points as points,
    reindex_to as reindex_to,
    selection as selection,
    unify_contexts as unify_contexts,
    universe as universe,
    variations as variations,
    weight as weight,
)
from .ambient import Rider as Rider, ambient_entries as ambient_entries  # isort: skip
from .explain import Explanation as Explanation, explain as explain  # isort: skip
from .kinds import Kind as Kind  # isort: skip
from .registration import vary as vary  # isort: skip
from .varied import Varied as Varied, member_of as member_of  # isort: skip
