"""Calendar helpers for monthly reporting periods."""
from __future__ import annotations

from datetime import date, timedelta


def month_start(day: date) -> date:
    return day.replace(day=1)


def month_end(day: date) -> date:
    """The last day of ``day``'s month."""
    following = date(day.year, day.month % 12 + 1, 1)
    return following - timedelta(days=1)


def add_months(day: date, months: int) -> date:
    """``day`` moved by ``months`` months, clamped to the target month's length."""
    index = day.year * 12 + day.month - 1 + months
    year, month = divmod(index, 12)
    first = date(year, month + 1, 1)
    return first.replace(day=min(day.day, month_end(first).day))


def months_between(start: date, end: date) -> list[date]:
    """First days of every month from ``start``'s month through ``end``'s month."""
    if end < start:
        return []
    months = []
    cursor = month_start(start)
    while cursor <= end:
        months.append(cursor)
        cursor = add_months(cursor, 1)
    return months


def quarter(day: date) -> int:
    return (day.month - 1) // 3 + 1


def fiscal_year(day: date, first_month: int = 1) -> int:
    """The fiscal year ``day`` falls in when years start in ``first_month``."""
    return day.year + (1 if first_month > 1 and day.month >= first_month else 0)
