"""Balanced journal entries and account balances."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from ledger.accounts import Chart
from ledger.money import Money, total


class UnbalancedEntry(ValueError):
    """The postings of an entry do not sum to zero."""


@dataclass(frozen=True)
class Posting:
    account: str
    amount: Money
    memo: str = ""


@dataclass
class Entry:
    posted: date
    narration: str
    postings: list[Posting] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)
    reference: str = ""

    def check(self) -> None:
        by_currency: dict[str, list[Money]] = defaultdict(list)
        for posting in self.postings:
            by_currency[posting.amount.currency].append(posting.amount)
        for currency, amounts in by_currency.items():
            if total(amounts, currency):
                raise UnbalancedEntry(f"{self.narration!r} is off by {total(amounts, currency)}")
        if len(self.postings) < 2:
            raise UnbalancedEntry(f"{self.narration!r} needs at least two postings")


class Journal:
    """Entries in posting order, with running balances per account."""

    def __init__(self, chart: Chart | None = None):
        self.chart = chart or Chart.standard()
        self.entries: list[Entry] = []

    def post(self, entry: Entry) -> Entry:
        entry.check()
        for posting in entry.postings:
            self.chart.open(posting.account, posting.amount.currency)
        self.entries.append(entry)
        return entry

    def transfer(self, posted: date, narration: str, source: str, target: str, amount: Money, **extra) -> Entry:
        return self.post(Entry(posted, narration, [Posting(target, amount), Posting(source, -amount)], **extra))

    def between(self, start: date, end: date) -> list[Entry]:
        return [entry for entry in self.entries if start <= entry.posted <= end]

    def balance(self, account: str, until: date | None = None, currency: str = "USD") -> Money:
        """Sum of postings to ``account`` and its descendants (debit positive)."""
        amounts = [posting.amount for entry in self.entries if until is None or entry.posted <= until
                   for posting in entry.postings
                   if (posting.account == account or posting.account.startswith(account + ":"))
                   and posting.amount.currency == currency]
        return total(amounts, currency)

    def balances(self, until: date | None = None, currency: str = "USD") -> dict[str, Money]:
        result: dict[str, Money] = {}
        for name in self.chart.names():
            amount = self.balance(name, until, currency)
            if amount:
                result[name] = amount
        return result

    def trial_balance(self, currency: str = "USD") -> Money:
        """Sum over root accounts; zero for a consistent journal."""
        return total((self.balance(root.name, currency=currency) for root in self.chart.roots()), currency)
