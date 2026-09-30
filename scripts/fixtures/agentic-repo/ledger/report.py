"""Monthly income and expense reports."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ledger.dates import month_end, month_start, months_between
from ledger.journal import Journal
from ledger.money import Money


@dataclass
class MonthRow:
    start: date
    end: date
    income: Money
    expenses: Money
    by_category: dict[str, Money] = field(default_factory=dict)

    @property
    def net(self) -> Money:
        return self.income - self.expenses


def month_row(journal: Journal, month: date, currency: str = "USD", depth: int = 2) -> MonthRow:
    start, end = month_start(month), month_end(month)
    income = Money.zero(currency)
    expenses = Money.zero(currency)
    categories: dict[str, Money] = {}
    for entry in journal.between(start, end):
        for posting in entry.postings:
            if posting.amount.currency != currency:
                continue
            if posting.account.startswith("income"):
                income = income - posting.amount
            elif posting.account.startswith("expenses"):
                expenses = expenses + posting.amount
                category = ":".join(posting.account.split(":")[: depth])
                categories[category] = categories.get(category, Money.zero(currency)) + posting.amount
    return MonthRow(start, end, income, expenses, dict(sorted(categories.items())))


def monthly(journal: Journal, start: date, end: date, currency: str = "USD") -> list[MonthRow]:
    return [month_row(journal, month, currency) for month in months_between(start, end)]


def render(rows: list[MonthRow], width: int = 12) -> str:
    lines = [f"{'month':<10}{'income':>{width}}{'expenses':>{width}}{'net':>{width}}"]
    for row in rows:
        lines.append(f"{row.start:%Y-%m}   {row.income.format():>{width}}{row.expenses.format():>{width}}"
                     f"{row.net.format():>{width}}")
        for category, amount in row.by_category.items():
            lines.append(f"  {category:<28}{amount.format():>{width}}")
    return "\n".join(lines)
