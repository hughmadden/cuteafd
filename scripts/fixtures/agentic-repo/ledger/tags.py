"""Hashtags in descriptions and memos: ``#travel``, ``#q3-offsite``."""
from __future__ import annotations

import re
from collections import Counter

TAG = re.compile(r"(?<![\w#])#([a-z0-9][a-z0-9_-]{0,31})", re.IGNORECASE)


def extract(text: str) -> tuple[str, ...]:
    """Tags in order of first appearance, lowercased, without duplicates."""
    seen: list[str] = []
    for match in TAG.finditer(text):
        tag = match.group(1).lower()
        if tag not in seen:
            seen.append(tag)
    return tuple(seen)


def strip(text: str) -> str:
    """``text`` without its tags and with whitespace collapsed."""
    return " ".join(TAG.sub("", text).split())


def histogram(texts) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        counts.update(extract(text))
    return counts


def rename(text: str, old: str, new: str) -> str:
    pattern = re.compile(rf"(?<![\w#])#{re.escape(old)}(?![\w-])", re.IGNORECASE)
    return pattern.sub(f"#{new}", text)
