"""First River Bank checking exports: ISO dates, signed amounts, a trailer line."""
from __future__ import annotations

from ledger.parser import Dialect, Transaction, parse_csv, split_csv_line

HEADER = ("Posting Date", "Details", "Amount", "Check or Slip #")


class FirstRiverImporter:
    name = "first-river"
    dialect = Dialect(date="Posting Date", description="Details", amount="Amount", reference="Check or Slip #")

    @staticmethod
    def accepts(header: str) -> bool:
        return tuple(field.strip() for field in split_csv_line(header)) == HEADER

    def load(self, text: str, account: str = "assets:checking") -> list[Transaction]:
        lines = [line for line in text.splitlines() if not line.startswith("Total")]
        return parse_csv("\n".join(lines), self.dialect, account)
