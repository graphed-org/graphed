"""The §1.1 variation-tag grammar: validation, canonicalization, and the two cap refusals.

A tag is the user-facing half of a variation label (`f"{name}_{tag}"`). Identifier tags (`up`,
`pdf_1`, the datacard p-form `2p5`) are kept verbatim; numeric spellings are canonicalized by
EXACT DECIMAL arithmetic to the e-form `m?\\d+(em\\d+)?`, so `"2"`, `"2.0"`, `"2e0"` and `"20e-1"`
all name one label and no IEEE round-trip artifact ever reaches a name.

Canonicalization works on (sign, digit string, power of ten) triples taken straight from the
input's characters, never on a float, and the 32-character cap is tested on the COMPUTED digit
count before any string is rendered — `"1e1000000000"` must refuse, not allocate a gigabyte.
"""

from __future__ import annotations

import re
from fractions import Fraction

from .errors import GraphedError

#: §1.1's tag-sanity bound on the RENDERED canonical tag (it bounds neither the label nor the
#: on-disk name).
MAX_TAG_CHARS = 32

_FLOAT_SUGAR = re.compile(r"(?P<sign>-?)(?P<int>\d+)(?:\.(?P<frac>\d+))?(?:[eE](?P<exp>[+-]?\d+))?\Z")
_CANONICAL = re.compile(r"(?P<sign>m?)(?P<int>\d+)(?:em(?P<exp>\d+))?\Z")
_P_FORM = re.compile(r"(?P<sign>m?)(?P<int>\d+)(?:p(?P<frac>\d+))?\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9_]+\Z")
#: spellings the float grammar deliberately excludes but which read as identifiers
_NOT_FINITE = frozenset({"inf", "infinity", "nan"})


def _normalize(negative: bool, digits: str, exp10: int) -> tuple[bool, str, int]:
    """Minimal mantissa: no leading zeros, no trailing zeros, exponent absorbing the shift."""
    digits = digits.lstrip("0")
    if not digits:
        return False, "0", 0  # negative zero canonicalizes to `0`, never `m0`
    while digits.endswith("0"):
        digits = digits[:-1]
        exp10 += 1
    return negative, digits, exp10


def _render(negative: bool, digits: str, exp10: int, source: str) -> str:
    if digits == "0":
        return "0"
    sign = 1 if negative else 0
    # the magnitude test runs on the COUNTED integer digits, so an absurd exponent refuses here
    # rather than materializing its rendering
    magnitude = max(len(digits) + exp10, 1)
    if magnitude > MAX_TAG_CHARS:
        raise GraphedError(
            f"variation tag {source!r}: its magnitude is {magnitude} digits, over the "
            f"{MAX_TAG_CHARS}-character bound"
        )
    length = magnitude + sign if exp10 >= 0 else len(digits) + 2 + len(str(-exp10)) + sign
    if length > MAX_TAG_CHARS:
        raise GraphedError(
            f"variation tag {source!r}: canonical tag length {length} exceeds the "
            f"{MAX_TAG_CHARS}-character cap"
        )
    marker = "m" if negative else ""
    if exp10 >= 0:
        return marker + digits + "0" * exp10
    return f"{marker}{digits}em{-exp10}"


def canonical_tag(tag: object) -> str:
    """The §1.1 tag a user spelling names, or a `GraphedError` saying which rule it broke."""
    if not isinstance(tag, str):
        raise GraphedError(
            f"variation tags must be strings, got {tag!r} — pass the spelling you want in the label"
        )
    if not tag:
        raise GraphedError("a variation tag must not be empty")
    if tag.lower() in _NOT_FINITE:
        raise GraphedError(f"variation tag {tag!r} does not name a finite value")
    sugar = _FLOAT_SUGAR.fullmatch(tag)
    if sugar is not None:
        frac = sugar["frac"] or ""
        exp10 = int(sugar["exp"] or 0) - len(frac)
        return _render(*_normalize(sugar["sign"] == "-", sugar["int"] + frac, exp10), tag)
    canonical = _CANONICAL.fullmatch(tag)
    if canonical is not None:  # a hand-typed e-form is re-rendered minimally, not passed through
        exp10 = -int(canonical["exp"] or 0)
        return _render(*_normalize(canonical["sign"] == "m", canonical["int"], exp10), tag)
    if not _IDENTIFIER.fullmatch(tag):
        raise GraphedError(
            f"variation tag {tag!r} is neither a numeric spelling nor an identifier "
            "([A-Za-z0-9_]+); labels must be usable as column and category names"
        )
    if tag[0].isdigit() and _P_FORM.fullmatch(tag) is None:
        # digit-leading identifiers read as numbers; only the datacard p-form is a legal one
        raise GraphedError(
            f"variation tag {tag!r} looks numeric but is not a valid spelling "
            "(no separators, no whitespace, no leading '+')"
        )
    return tag


def numeric_value(tag: str) -> Fraction | None:
    """The exact value a canonical or p-encoded tag names, `None` for a non-numeric identifier.

    The p-form deliberately does not canonicalize, so it is the one residual duplicate class:
    §1.1's family check compares these values to reject `{"0.5", "0p5"}` naming one universe twice.
    """
    canonical = _CANONICAL.fullmatch(tag)
    if canonical is not None:
        value = Fraction(int(canonical["int"]), 10 ** int(canonical["exp"] or 0))
        return -value if canonical["sign"] == "m" else value
    p_form = _P_FORM.fullmatch(tag)
    if p_form is not None:
        frac = p_form["frac"] or ""
        value = Fraction(int(p_form["int"] + frac), 10 ** len(frac))
        return -value if p_form["sign"] == "m" else value
    return None
