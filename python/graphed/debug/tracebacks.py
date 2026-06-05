"""`format_traceback` — render a `StageError` as a user-source traceback (plan M6).

The output collapses `graphed*` internal frames and shows only the user's analysis frames, with the
failing line last (Python's "most recent call last" convention) and clearly marked. Because it is
built from the `StageError`'s carried `SourceFrame` data — not a live traceback object — it renders
identically after the error has crossed a process boundary.
"""

from __future__ import annotations

from .errors import StageError


def format_traceback(err: StageError) -> str:
    """Format ``err`` as a user-source-mapped traceback string."""
    lines = ["Traceback (most recent call last) — user analysis frames:"]
    # oldest ancestor first, failing line last
    chain = list(reversed(err.frames))
    for i, fr in enumerate(chain):
        marker = "  --> " if i == len(chain) - 1 else "      "
        func = fr.function or "<module>"
        lines.append(f'{marker}File "{fr.filename}", line {fr.lineno}, in {func}')
        if fr.source:
            lines.append(f"          {fr.source}")
    lines.append(
        f"{err.cause_type}: {err.cause_message}  "
        f"[stage op {err.op!r}, partition {err.partition}, opt_level={err.opt_level}]"
    )
    if err.input_forms:
        lines.append(f"  input forms: {', '.join(err.input_forms)}")
    return "\n".join(lines)
