"""``format_traceback`` — render a ``StageError`` as a user-source traceback (plan M6).

The output collapses ``graphed*`` internal frames and shows only the user's analysis frames, with
the failing line last (Python's "most recent call last" convention) and clearly marked. Because it
is built from the ``StageError``'s carried ``SourceFrame`` data — not a live traceback object — it
renders identically after the error has crossed a process boundary.
"""

from __future__ import annotations

from .errors import StageError


def format_traceback(err: StageError) -> str:
    """Format ``err`` as a user-source-mapped traceback string.

    >>> import awkward as ak
    >>> import graphed, graphed.debug as gd
    >>> from graphed.awkward import from_awkward
    >>> def calibrate(x):
    ...     raise ValueError("no calibration table")
    >>> x = from_awkward("x", ak.Array([1.0, 2.0]))
    >>> try:
    ...     graphed.materialize(x.map(calibrate, name="calibrate", output_type="float64"))
    ... except gd.StageError as err:
    ...     print(gd.format_traceback(err))
    Traceback (most recent call last) — user analysis frames:
      --> File "<doctest ...>", line 2, in <module>
              x.map(calibrate, name='calibrate', output_type='float64')
    ...
    ValueError: no calibration table  [stage op "map 'calibrate'", ...]
    ...
    """
    return f"Traceback (most recent call last) — user analysis frames:\n{err.view()}"
