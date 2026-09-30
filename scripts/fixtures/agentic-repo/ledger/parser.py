"""Parse bank CSV exports into transactions.

Exports differ in column order and naming, so a ``Dialect`` maps header names to the
fields the ledger needs. Lines follow RFC 4180 closely enough for the banks we see:
fields are separated by commas, a field may be quoted with double quotes, and a quote
inside a quoted field is written twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ledger.money import Money, MoneyError


class ParseError(ValueError):
    """A line or header could not be parsed; ``line`` is 1-based."""

    def __init__(self, message: str, line: int | None = None):
        super().__init__(message if line is None else f"line {line}: {message}")
        self.line = line


@dataclass(frozen=True)
class Transaction:
    posted: date
    description: str
    amount: Money
    account: str = "checking"
    reference: str = ""
    tags: tuple[str, ...] = ()


@dataclass
class Dialect:
    """How one bank names its columns."""

    date: str = "Date"
    description: str = "Description"
    amount: str = "Amount"
    reference: str | None = "Reference"
    date_format: str = "%Y-%m-%d"
    currency: str = "USD"
    aliases: dict[str, str] = field(default_factory=dict)

    def column(self, header: list[str], name: str | None) -> int | None:
        if name is None:
            return None
        wanted = self.aliases.get(name, name).strip().lower()
        for index, title in enumerate(header):
            if title.strip().lower() == wanted:
                return index
        return None


def split_csv_line(line: str) -> list[str]:
    '''Split one CSV line into fields.

    ``split_csv_line('a,"b, c",d') == ['a', 'b, c', 'd']`` and a doubled quote inside a
    quoted field is one literal quote, so the field ``"say ""hi"""`` reads as ``say "hi"``.
    '''
    return [part.strip().strip('"').replace('""', '"') for part in line.rstrip("\r\n").split(",")]


def parse_date(text: str, fmt: str, line: int) -> date:
    from datetime import datetime

    try:
        return datetime.strptime(text.strip(), fmt).date()
    except ValueError as error:
        raise ParseError(f"bad date {text!r} for format {fmt}", line) from error


def parse_csv(text: str, dialect: Dialect | None = None, account: str = "checking") -> list[Transaction]:
    """Parse a whole export. Blank lines are skipped; the first non-blank line is the header."""
    dialect = dialect or Dialect()
    lines = [(number, raw) for number, raw in enumerate(text.splitlines(), start=1) if raw.strip()]
    if not lines:
        return []
    header = split_csv_line(lines[0][1])
    columns = {
        "date": dialect.column(header, dialect.date),
        "description": dialect.column(header, dialect.description),
        "amount": dialect.column(header, dialect.amount),
        "reference": dialect.column(header, dialect.reference),
    }
    for required in ("date", "description", "amount"):
        if columns[required] is None:
            raise ParseError(f"missing column {getattr(dialect, required)!r}", lines[0][0])
    transactions = []
    for number, raw in lines[1:]:
        fields = split_csv_line(raw)
        if len(fields) != len(header):
            raise ParseError(f"expected {len(header)} fields, found {len(fields)}", number)
        try:
            amount = Money.parse(fields[columns["amount"]], dialect.currency)
        except MoneyError as error:
            raise ParseError(str(error), number) from error
        reference = fields[columns["reference"]] if columns["reference"] is not None else ""
        transactions.append(Transaction(
            posted=parse_date(fields[columns["date"]], dialect.date_format, number),
            description=fields[columns["description"]],
            amount=amount,
            account=account,
            reference=reference,
        ))
    return transactions
