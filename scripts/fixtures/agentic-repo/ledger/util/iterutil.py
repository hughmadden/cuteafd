"""Iteration helpers."""
from __future__ import annotations

from itertools import islice
from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")
K = TypeVar("K")


def chunked(items: Iterable[T], size: int) -> Iterator[list[T]]:
    if size <= 0:
        raise ValueError("size must be positive")
    iterator = iter(items)
    while chunk := list(islice(iterator, size)):
        yield chunk


def group_by(items: Iterable[T], key: Callable[[T], K]) -> dict[K, list[T]]:
    groups: dict[K, list[T]] = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return groups


def pairwise(items: Iterable[T]) -> Iterator[tuple[T, T]]:
    iterator = iter(items)
    previous = next(iterator, None)
    for item in iterator:
        yield previous, item
        previous = item


def unique(items: Iterable[T], key: Callable[[T], object] = lambda item: item) -> list[T]:
    seen, result = set(), []
    for item in items:
        marker = key(item)
        if marker not in seen:
            seen.add(marker)
            result.append(item)
    return result
