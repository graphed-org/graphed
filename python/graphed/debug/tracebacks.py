"""``format_traceback`` — render a ``StageError`` as a user-source traceback (plan M6).

The output collapses ``graphed*`` internal frames and shows only the user's analysis frames, with
the failing line last (Python's "most recent call last" convention) and clearly marked. Because it
is built from the ``StageError``'s carried ``SourceFrame`` data — not a live traceback object — it
renders identically after the error has crossed a process boundary.
"""

from __future__ import annotations

from .errors import StageError


def format_traceback(err: StageError) -> str:
    """Format ``err`` as a user-source-mapped traceback string."""
    return f"Traceback (most recent call last) — user analysis frames:\n{err.view()}"
