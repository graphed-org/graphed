"""#155: the top-level namespace lists user verbs only; moved names warn for one release."""

from __future__ import annotations

import importlib
import inspect
import warnings

import pytest

import graphed

PUBLIC = {
    "Array",
    "GraphedError",
    "GraphedTypeError",
    "GraphedWarning",
    "OutputTypeError",
    "PlanPieces",
    "PointError",
    "Session",
    "Varied",
    "WorkerLostError",
    "aggregate_plan",
    "apply",
    "collate",
    "compile_ir",
    "current_session",
    "evaluate_ir",
    "expand",
    "head",
    "join",
    "join_plan",
    "materialize",
    "new_session",
    "raise_on_floating_point",
    "read_columns",
    "repartition",
    "set_default_backend",
    "set_session",
    "shuffle_plan",
    "vary",
}
HOMES = {
    "graphed.systematics": "Explanation Kind broadcast_like context_of explain explain_variations "
    "impact_by_label labels member_of nominal points read_columns_by_label reindex_to selection "
    "unify_contexts universe variations weight",
    "graphed.systematics.varied": "broadcasting expanding",
    "graphed.provenance": "Provenance capture is_enabled set_enabled",
    "graphed.projection": "CONSERVATIVE BufferNeed BufferProjection OnFail Projection "
    "ProjectionError handle_opaque",
    "graphed.backend": "Backend Form ParamValue",
    "graphed.execute": "CompiledGraph refuse_chunk_partials",
    "graphed.aggregate": "resolve_backend",
    "graphed.array": "BoundMethod",
    "graphed.shuffle": "pack_key",
}
MOVED = {name: home for home, names in HOMES.items() for name in names.split()}
PACKAGES = (
    "graphed",
    "graphed.numpy",
    "graphed.core",
    "graphed.awkward",
    "graphed.debug",
    "graphed.checkpoint",
)


def test_the_top_level_lists_the_user_surface() -> None:
    assert set(graphed.__all__) == PUBLIC
    assert all(hasattr(graphed, name) for name in PUBLIC)


@pytest.mark.parametrize("package", PACKAGES)
def test_dir_is_the_public_names(package: str) -> None:
    module = importlib.import_module(package)
    assert dir(module) == sorted(module.__all__)
    assert module.__dir__() is not module.__all__


@pytest.mark.parametrize(("name", "home"), sorted(MOVED.items()))
def test_a_moved_name_resolves_from_its_home_with_a_warning(name: str, home: str) -> None:
    with pytest.warns(DeprecationWarning, match=rf"graphed\.{name} moved to {home}\.{name}"):
        value = getattr(graphed, name)
    assert value is getattr(importlib.import_module(home), name)
    assert name not in graphed.__all__


def test_an_unknown_name_raises_without_a_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AttributeError, match="no attribute 'no_such_verb'"):
            graphed.no_such_verb  # noqa: B018


def test_a_moved_array_verb_keeps_its_varied_disposition() -> None:
    """m48's guard walks `graphed.__all__`, so it no longer reaches the verbs that moved out."""
    moved_verbs = set()
    for name, home in MOVED.items():
        member = getattr(importlib.import_module(home), name)
        if inspect.isfunction(member) and any(
            "Array" in str(p.annotation) for p in inspect.signature(member).parameters.values()
        ):
            moved_verbs.add(name)
    assert moved_verbs
    assert moved_verbs <= set(graphed.VERB_DISPOSITIONS)


def test_the_pre_move_module_paths_load_with_the_package() -> None:
    assert graphed.by_label.impact_by_label is graphed.systematics.impact_by_label
    assert graphed.accessors.labels is graphed.systematics.labels
    assert graphed._kinds.Kind is graphed.systematics.Kind
