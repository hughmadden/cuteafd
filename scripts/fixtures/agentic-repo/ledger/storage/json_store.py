"""Store a journal as JSON (amounts as integer cents)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ledger.journal import Entry, Journal, Posting
from ledger.money import Money

FORMAT = 2


def dump(journal: Journal) -> str:
    entries = [{
        "posted": entry.posted.isoformat(),
        "narration": entry.narration,
        "reference": entry.reference,
        "tags": sorted(entry.tags),
        "postings": [{"account": p.account, "cents": p.amount.cents, "currency": p.amount.currency, "memo": p.memo}
                     for p in entry.postings],
    } for entry in journal.entries]
    return json.dumps({"format": FORMAT, "entries": entries}, indent=2, sort_keys=True)


def load(text: str) -> Journal:
    data = json.loads(text)
    if data.get("format") not in (1, FORMAT):
        raise ValueError(f"unsupported journal format {data.get('format')!r}")
    journal = Journal()
    for item in data["entries"]:
        postings = [Posting(p["account"], Money(p["cents"], p.get("currency", "USD")), p.get("memo", ""))
                    for p in item["postings"]]
        journal.post(Entry(date.fromisoformat(item["posted"]), item["narration"], postings,
                           set(item.get("tags", [])), item.get("reference", "")))
    return journal


def save(journal: Journal, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(dump(journal))
    temporary.replace(path)
