"""Mask personal data in complaint text before it reaches a model, a log or an LLM.

Rule-based on purpose: predictable, fast and easy to audit. It covers the
identifiers a bank complaint typically contains. Names and street addresses
need an NER model and are out of scope here.
"""

import re

PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Card numbers: 13 to 19 digits, optionally grouped by spaces or dashes; Luhn-checked below.
    ("CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)")),
    # Long digit runs left after the checks above: account, loan or routing numbers.
    ("ACCOUNT", re.compile(r"\b\d{8,17}\b")),
]


def luhn_ok(digits):
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def mask(text):
    """Return (masked_text, counts) where counts maps PII type to occurrences."""
    counts = {}

    def replace(kind):
        def _sub(match):
            if kind == "CARD":
                digits = re.sub(r"\D", "", match.group())
                if not luhn_ok(digits):
                    return match.group()  # leave for the ACCOUNT pattern
            counts[kind] = counts.get(kind, 0) + 1
            return f"[{kind}]"
        return _sub

    for kind, pattern in PATTERNS:
        text = pattern.sub(replace(kind), text)
    return text, counts
