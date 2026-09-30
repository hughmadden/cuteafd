"""Exchange rates: providers and a time-bounded cache in front of them.

A provider answers ``rate(base, quote)`` as an exact fraction ``(numerator,
denominator)``. ``RateCache`` remembers answers for ``ttl`` seconds of the injected
clock, so tests can move time without sleeping.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol


class RateProvider(Protocol):
    def rate(self, base: str, quote: str) -> tuple[int, int]: ...


class RateError(LookupError):
    """No rate is known for a currency pair."""


@dataclass
class StaticRates:
    """Fixed rates, e.g. from a config file; inverse pairs are derived."""

    table: dict[tuple[str, str], tuple[int, int]] = field(default_factory=dict)
    calls: int = 0

    def rate(self, base: str, quote: str) -> tuple[int, int]:
        self.calls += 1
        if base == quote:
            return (1, 1)
        if (base, quote) in self.table:
            return self.table[(base, quote)]
        if (quote, base) in self.table:
            numerator, denominator = self.table[(quote, base)]
            return (denominator, numerator)
        raise RateError(f"no rate for {base}/{quote}")


@dataclass
class _Cached:
    value: tuple[int, int]
    fetched_at: float


class RateCache:
    """Caches a provider's rates for ``ttl`` seconds."""

    def __init__(self, provider: RateProvider, ttl: float = 3600.0, clock: Callable[[], float] = time.monotonic):
        self.provider = provider
        self.ttl = ttl
        self.clock = clock
        self._entries: dict[tuple[str, str], _Cached] = {}
        self.hits = 0
        self.misses = 0

    def get(self, base: str, quote: str) -> tuple[int, int]:
        key = (base, quote)
        now = self.clock()
        cached = self._entries.get(key)
        if cached is not None and now - cached.fetched_at > self.ttl:
            self.hits += 1
            return cached.value
        self.misses += 1
        value = self.provider.rate(base, quote)
        self._entries[key] = _Cached(value, now)
        return value

    def invalidate(self, base: str | None = None) -> int:
        """Drop cached rates (all, or those with ``base``); returns how many were dropped."""
        if base is None:
            dropped = len(self._entries)
            self._entries.clear()
            return dropped
        keys = [key for key in self._entries if key[0] == base]
        for key in keys:
            del self._entries[key]
        return len(keys)
