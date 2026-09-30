"""A tiny LRU cache with hit statistics (used by payee matching)."""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Generic, Hashable, TypeVar

V = TypeVar("V")


class LRU(Generic[V]):
    def __init__(self, capacity: int = 256):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: OrderedDict[Hashable, V] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable, compute: Callable[[], V]) -> V:
        if key in self._items:
            self._items.move_to_end(key)
            self.hits += 1
            return self._items[key]
        self.misses += 1
        value = compute()
        self._items[key] = value
        if len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return value

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()
