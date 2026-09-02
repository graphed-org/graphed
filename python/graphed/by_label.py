"""The two m49 per-label introspection verbs (§3.4, §5.3) and the operand form they share.

Both answer over the SAME operand — `Sequence[Varied] | Mapping[str, Sequence[Array]]`, the
labelled analogue of `read_columns`' first argument — and both range over the §2.4 label UNION
resolved by `graphed.member_of`, never the strict `graphed.universe`, which raises `KeyError` on a
heterogeneous operand (a jes-varied kinematic beside a btag-varied weight is the corpus mainline).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .array import Array
from .errors import GraphedError
from .projection import read_columns
from .session import Session
from .varied import Varied, member_of, union_labels

__all__ = ["cone", "impact_by_label", "read_columns_by_label"]

#: §3.4's operand, for the bodies below. NOT what the two verbs annotate: under
#: `from __future__ import annotations` an alias stringifies to its own bare name, hiding the
#: `Array` mention §2.3d's discovery filter reads, so both verbs spell this union INLINE.
Outputs = Sequence[Varied] | Mapping[str, Sequence[Array]]


def cone(session: Session, node_id: int) -> set[int]:
    """Every record-time node id reachable from `node_id`, via the generic `session.walk`."""
    seen: set[int] = set()

    def note(nid: int, *_rest: object) -> None:
        seen.add(nid)

    session.walk(Array(session, node_id), source=note, op=note, external=note)
    return seen


def _per_label(verb: str, outputs: Outputs) -> dict[str, list[Array]]:
    """Resolve either operand form to `{label: that label's output Arrays}` (§3.4).

    The two forms do not mix, and the rejection names the offending element's TYPE — an unchecked
    operand instead reaches for `.session` on whatever it was handed and dies with an
    `AttributeError`.
    """
    if isinstance(outputs, Mapping):
        resolved: dict[str, list[Array]] = {}
        for label, arrays in outputs.items():
            if not _is_sequence(arrays):
                raise GraphedError(
                    f"{verb}: label {label!r} maps to {type(arrays).__name__}; a mapping operand "
                    "maps each label to a SEQUENCE of Array"
                )
            for item in arrays:
                if not isinstance(item, Array):
                    raise GraphedError(
                        f"{verb}: label {label!r} maps to a sequence containing "
                        f"{type(item).__name__}; a mapping operand maps each label to a sequence "
                        "of Array — pass a sequence of Varied instead to have the labels resolved"
                    )
            resolved[label] = list(arrays)
        return resolved
    if not _is_sequence(outputs):
        raise GraphedError(
            f"{verb} takes a sequence of Varied or a mapping of label to a sequence of Array, "
            f"not {type(outputs).__name__}"
        )
    containers: list[Varied] = []
    for entry in outputs:
        if not isinstance(entry, Varied):
            raise GraphedError(
                f"{verb}: a sequence operand must be all Varied, got {type(entry).__name__}; a "
                "plain Array carries no label attribution"
            )
        for label, universe in entry._members.items():
            if isinstance(universe, Varied):
                # §2.2 admits nested members on a registered weight factor; the per-label walk
                # cannot resolve past one level, so it says so rather than reaching through.
                raise GraphedError(
                    f"{verb}: the {label!r} member of a Varied is itself a "
                    f"{type(universe).__name__}; resolve the nesting first"
                )
        containers.append(entry)
    return {
        label: [member_of(container, label) for container in containers]
        for label in union_labels(*containers)
    }


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def impact_by_label(outputs: Sequence[Varied] | Mapping[str, Sequence[Array]]) -> dict[str, tuple[int, ...]]:
    """§3.4: per label, the record node ids that label reaches and `"nominal"` does not, sorted.

    The reachability DIFFERENCE, not an id watermark: interleaved broadcast recording makes
    watermark bracketing order-dependent, and a node two non-nominal labels share appears in BOTH
    impact sets. No `source_nid` — `session.walk` is source-agnostic.
    """
    per_label = _per_label("graphed.impact_by_label", outputs)
    central = _reach(per_label.get("nominal", []))
    # `nominal` is `central - central`: answer it from the walk already done rather than repeating it
    return {
        label: () if label == "nominal" else tuple(sorted(_reach(arrays) - central))
        for label, arrays in per_label.items()
    }


def _reach(arrays: Sequence[Array]) -> set[int]:
    return {nid for array in arrays for nid in cone(array.session, array.node_id)}


def read_columns_by_label(
    outputs: Sequence[Varied] | Mapping[str, Sequence[Array]], source_nid: int
) -> dict[str, tuple[str, ...] | None]:
    """§5.3: per label, that label's sorted read set on source `source_nid`, making the read-width
    cost of a shift visible. `None` is `read_columns`' own conservative answer — "read every
    column" — and must never merge with `()`, which says the opposite.
    """
    per_label = _per_label("graphed.read_columns_by_label", outputs)
    return {label: read_columns(arrays, source_nid) for label, arrays in per_label.items()}
