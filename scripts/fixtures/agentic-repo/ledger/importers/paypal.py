"""PayPal activity exports: a currency column, gross/fee/net amounts, status filter."""
from __future__ import annotations

from ledger.money import Money
from ledger.parser import Dialect, ParseError, Transaction, parse_date, split_csv_line


class PaypalImporter:
    name = "paypal"
    dialect = Dialect(date="Date", description="Name", amount="Net", reference="Transaction ID",
                      date_format="%d/%m/%Y")

    @staticmethod
    def accepts(header: str) -> bool:
        fields = {field.strip() for field in split_csv_line(header)}
        return {"Gross", "Fee", "Net", "Transaction ID"} <= fields

    def load(self, text: str, account: str = "assets:paypal") -> list[Transaction]:
        lines = [line for line in text.splitlines() if line.strip()]
        header = [field.strip() for field in split_csv_line(lines[0])]
        index = {name: position for position, name in enumerate(header)}
        for column in ("Date", "Name", "Net", "Currency", "Status", "Transaction ID"):
            if column not in index:
                raise ParseError(f"missing column {column!r}", 1)
        transactions = []
        for number, line in enumerate(lines[1:], start=2):
            fields = split_csv_line(line)
            if fields[index["Status"]] != "Completed":
                continue
            currency = fields[index["Currency"]]
            transactions.append(Transaction(
                posted=parse_date(fields[index["Date"]], self.dialect.date_format, number),
                description=fields[index["Name"]] or "PayPal",
                amount=Money.parse(fields[index["Net"]], currency),
                account=account,
                reference=fields[index["Transaction ID"]],
            ))
        return transactions
