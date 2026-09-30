"""Recurring transactions and a naive cash forecast."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ledger.dates import add_months
from ledger.money import Money


@dataclass(frozen=True)
class Recurring:
    description: str
    amount: Money
    first: date
    every_months: int = 1
    count: int | None = None

    def occurrences(self, until: date) -> list[date]:
        days, index = [], 0
        while True:
            day = add_months(self.first, index * self.every_months)
            if day > until or (self.count is not None and index >= self.count):
                return days
            days.append(day)
            index += 1


def forecast(opening: Money, items: list[Recurring], start: date, until: date) -> list[tuple[date, Money]]:
    """Running balance after each recurring item between ``start`` and ``until``."""
    events = sorted((day, item.description, item.amount) for item in items
                    for day in item.occurrences(until) if day >= start)
    balance, points = opening, []
    for day, _, amount in events:
        balance = balance + amount
        points.append((day, balance))
    return points


def lowest(points: list[tuple[date, Money]]) -> tuple[date, Money] | None:
    return min(points, key=lambda point: point[1].cents) if points else None
