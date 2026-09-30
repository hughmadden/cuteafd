"""Envelope budgets: a monthly allowance per expense category, with rollover."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ledger.money import Money
from ledger.report import month_row
from ledger.journal import Journal


@dataclass
class Envelope:
    category: str
    allowance: Money
    rollover: bool = True


@dataclass
class EnvelopeStatus:
    category: str
    available: Money
    spent: Money

    @property
    def remaining(self) -> Money:
        return self.available - self.spent

    @property
    def overspent(self) -> bool:
        return self.remaining.cents < 0


@dataclass
class Budget:
    envelopes: list[Envelope] = field(default_factory=list)

    def add(self, category: str, allowance: Money, rollover: bool = True) -> Envelope:
        envelope = Envelope(category, allowance, rollover)
        self.envelopes.append(envelope)
        return envelope

    def status(self, journal: Journal, months: list[date]) -> list[EnvelopeStatus]:
        """Status after the last of ``months``, carrying unspent allowance where allowed."""
        carried = {envelope.category: Money.zero(envelope.allowance.currency) for envelope in self.envelopes}
        spent_last = dict(carried)
        for month in months:
            row = month_row(journal, month)
            for envelope in self.envelopes:
                spent = row.by_category.get(envelope.category, Money.zero(envelope.allowance.currency))
                available = envelope.allowance + (carried[envelope.category] if envelope.rollover else Money.zero())
                carried[envelope.category] = available - spent
                spent_last[envelope.category] = spent
        return [EnvelopeStatus(e.category, carried[e.category] + spent_last[e.category], spent_last[e.category])
                for e in self.envelopes]
