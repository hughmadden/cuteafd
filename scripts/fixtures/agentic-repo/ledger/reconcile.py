"""Match imported transactions against journal entries by date window and amount."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from ledger.journal import Journal
from ledger.parser import Transaction


@dataclass
class Match:
    transaction: Transaction
    entry_index: int
    days_apart: int


@dataclass
class Reconciliation:
    matched: list[Match]
    unmatched_transactions: list[Transaction]
    unmatched_entries: list[int]

    @property
    def clean(self) -> bool:
        return not self.unmatched_transactions and not self.unmatched_entries


def reconcile(journal: Journal, transactions: list[Transaction], account: str, window_days: int = 3) -> Reconciliation:
    """Greedy one-to-one matching: closest date first, then earliest entry."""
    candidates = [
        (index, entry) for index, entry in enumerate(journal.entries)
        if any(posting.account == account for posting in entry.postings)
    ]
    used: set[int] = set()
    matched: list[Match] = []
    unmatched: list[Transaction] = []
    for transaction in sorted(transactions, key=lambda t: (t.posted, t.description)):
        best = None
        for index, entry in candidates:
            if index in used:
                continue
            amount = next(p.amount for p in entry.postings if p.account == account)
            if amount != transaction.amount:
                continue
            apart = abs((entry.posted - transaction.posted).days)
            if apart > window_days:
                continue
            if best is None or (apart, index) < (best[1], best[0]):
                best = (index, apart)
        if best is None:
            unmatched.append(transaction)
        else:
            used.add(best[0])
            matched.append(Match(transaction, best[0], best[1]))
    leftovers = [index for index, _ in candidates if index not in used]
    return Reconciliation(matched, unmatched, leftovers)


def window(days: int) -> timedelta:
    return timedelta(days=days)
