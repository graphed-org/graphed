"""How a nuisance's tag was registered on an event context (§9.1's kind vocabulary).

A `Flag`, so a registration that is several things at once is their UNION rather than a third
word: the name-identity idiom registers one nuisance as a shift of a collection AND as a weight
factor, and reports `Kind.WEIGHT | Kind.SHIFT`. A treatment added later composes with these the
same way.
"""

from __future__ import annotations

import enum


class Kind(enum.Flag):
    #: registered by the weight form: a per-event factor in the context's ambient weight
    WEIGHT = enum.auto()
    #: registered by the shift form: a collection replaced by its varied members
    SHIFT = enum.auto()

    def __repr__(self) -> str:
        return f"Kind.{self.name}"
