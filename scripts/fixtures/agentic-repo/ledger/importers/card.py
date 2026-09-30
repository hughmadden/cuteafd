"""North Card credit card exports: US dates, charges positive (so the sign flips)."""
from __future__ import annotations

from dataclasses import replace

from ledger.parser import Dialect, Transaction, parse_csv, split_csv_line


class NorthCardImporter:
    name = "north-card"
    dialect = Dialect(date="Trans. Date", description="Description", amount="Amount", reference=None,
                      date_format="%m/%d/%Y")

    @staticmethod
    def accepts(header: str) -> bool:
        fields = [field.strip() for field in split_csv_line(header)]
        return fields[:2] == ["Trans. Date", "Post Date"]

    def load(self, text: str, account: str = "liabilities:card") -> list[Transaction]:
        transactions = parse_csv(text, self.dialect, account)
        return [replace(transaction, amount=-transaction.amount) for transaction in transactions]
