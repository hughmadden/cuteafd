"""Write postings as CSV, quoting fields the way ``ledger.parser`` reads them."""
from __future__ import annotations

from ledger.journal import Journal


def quote(field: str) -> str:
    if any(ch in field for ch in ',"\n'):
        return '"' + field.replace('"', '""') + '"'
    return field


def postings_csv(journal: Journal) -> str:
    lines = ["Date,Description,Account,Amount,Currency"]
    for entry in journal.entries:
        for posting in entry.postings:
            amount = posting.amount.format(symbol=False).replace(",", "")
            lines.append(",".join(quote(field) for field in (entry.posted.isoformat(), entry.narration,
                                                             posting.account, amount, posting.amount.currency)))
    return "\n".join(lines) + "\n"
