"""§2.3c: every DISCOVERED public gak function carries a `Varied` classification.

The enumeration is dynamic at test time, not a literal name tuple — which would let a future gak
function go silently unclassified — and it is self-repairing: a new function is fixed in `src`,
never by editing this frozen file. gak defines no `__all__`, so a discovery step that returned an
empty or wrong set would pass tautologically; §2.3c's floor in this same test is what stops that.
"""

from __future__ import annotations

import inspect

import graphed.awkward as ga
from graphed.awkward import functions as gak_module
from graphed.awkward import gak

#: §2.3c's class set for the gak layer (distinct from §2.3d's module-verb vocabulary)
CLASSES = {"broadcast", "container-traversing", "tuple-returning", "eager-metadata", "refusing"}
#: freeze-time count under the binding discovery rule below
GAK_FLOOR = 65
#: one NAMED member per class, so a discovery set that collapsed to one class cannot pass
NAMED = {
    "broadcast": "where",
    "container-traversing": "zip",
    "tuple-returning": "unzip",
    "eager-metadata": "fields",
    "refusing": "join",
}


def _discover() -> set[str]:
    """The MODULE is named, not the package: `graphed.awkward` re-exports modules and classes only.
    The `__module__` filter is what keeps the re-exported payload descriptors out."""
    return {
        name
        for name, member in inspect.getmembers(gak_module, inspect.isfunction)
        if member.__module__ == "graphed.awkward.functions" and not name.startswith("_")
    }


def test_the_discovery_rule_reads_the_module_gak_actually_is() -> None:
    assert gak is gak_module
    assert getattr(ga, "num", None) is None  # the PACKAGE discovers no gak function
    assert not hasattr(gak_module, "__all__")  # which is why the floor below is load-bearing


def test_every_discovered_gak_function_carries_a_classification() -> None:
    discovered = _discover()
    unclassified = sorted(discovered - set(gak_module.GAK_DISPOSITIONS))
    assert not unclassified, f"{unclassified} would fall through gak's dispatch onto a Varied"
    assert {gak_module.GAK_DISPOSITIONS[name] for name in discovered} <= CLASSES


def test_the_non_vacuity_floor() -> None:
    discovered = _discover()
    assert len(discovered) >= GAK_FLOOR
    classified = {name: gak_module.GAK_DISPOSITIONS[name] for name in discovered}
    for expected_class, representative in NAMED.items():
        assert classified[representative] == expected_class
