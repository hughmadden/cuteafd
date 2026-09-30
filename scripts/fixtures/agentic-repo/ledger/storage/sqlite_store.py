"""Store a journal in SQLite (one table of entries, one of postings)."""
from __future__ import annotations

import sqlite3
from datetime import date

from ledger.journal import Entry, Journal, Posting
from ledger.money import Money

SCHEMA = """
create table if not exists entries (id integer primary key, posted text not null, narration text not null,
                                    reference text not null default '');
create table if not exists postings (entry integer not null references entries(id), account text not null,
                                     cents integer not null, currency text not null, memo text not null default '');
create index if not exists postings_account on postings(account);
"""


def connect(path: str = ":memory:") -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    return connection


def write(connection: sqlite3.Connection, journal: Journal) -> int:
    with connection:
        connection.execute("delete from postings")
        connection.execute("delete from entries")
        for entry in journal.entries:
            cursor = connection.execute("insert into entries (posted, narration, reference) values (?, ?, ?)",
                                        (entry.posted.isoformat(), entry.narration, entry.reference))
            connection.executemany(
                "insert into postings (entry, account, cents, currency, memo) values (?, ?, ?, ?, ?)",
                [(cursor.lastrowid, p.account, p.amount.cents, p.amount.currency, p.memo) for p in entry.postings])
    return len(journal.entries)


def read(connection: sqlite3.Connection) -> Journal:
    journal = Journal()
    for entry_id, posted, narration, reference in connection.execute(
            "select id, posted, narration, reference from entries order by id"):
        postings = [Posting(account, Money(cents, currency), memo) for account, cents, currency, memo in
                    connection.execute("select account, cents, currency, memo from postings where entry = ?",
                                       (entry_id,))]
        journal.post(Entry(date.fromisoformat(posted), narration, postings, reference=reference))
    return journal
