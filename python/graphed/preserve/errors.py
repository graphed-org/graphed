"""Errors raised by the preservation layer (plan M9)."""

from __future__ import annotations


class PreserveError(Exception):
    """Base class for preservation errors."""


class UnresolvedPayload(PreserveError):
    """A referenced payload (IR / dataset / correction / model / sourcemap) is not in the archive.

    Reproduction fails loudly with the missing content hash — never a silent wrong result (plan M9)."""

    def __init__(self, content_hash: str, *, what: str = "payload") -> None:
        self.content_hash = content_hash
        super().__init__(f"unresolved {what} {content_hash}")
