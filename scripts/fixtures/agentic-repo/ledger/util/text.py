"""Text normalization for descriptions and payee matching."""
from __future__ import annotations

import re
import unicodedata

NOISE = re.compile(r"\b(pos|debit|purchase|card\s*\d{4}|ref\s*#?\s*\w+)\b", re.IGNORECASE)
SPACES = re.compile(r"\s+")


def fold(text: str) -> str:
    """Lowercase, strip accents and collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return SPACES.sub(" ", stripped).strip().lower()


def payee(description: str) -> str:
    """A description with card-network noise removed, for grouping by payee."""
    return SPACES.sub(" ", NOISE.sub(" ", fold(description))).strip()


def truncate(text: str, width: int, ellipsis: str = "…") -> str:
    if width <= 0:
        return ""
    return text if len(text) <= width else text[: max(0, width - len(ellipsis))] + ellipsis


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", fold(text)).strip("-")
