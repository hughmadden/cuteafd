"""Command line: ``python -m ledger.cli import|report|balances``."""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from ledger.journal import Journal
from ledger.parser import Dialect, parse_csv
from ledger.report import monthly, render
from ledger.rules import RuleSet


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ledger")
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="import a bank CSV export and print the postings")
    imp.add_argument("csv", type=Path)
    imp.add_argument("--account", default="assets:checking")
    imp.add_argument("--rules", type=Path, help="categorization rules (one 'pattern => account' per line)")
    imp.add_argument("--currency", default="USD")
    rep = sub.add_parser("report", help="monthly income and expenses")
    rep.add_argument("csv", type=Path)
    rep.add_argument("--start", type=date.fromisoformat, required=True)
    rep.add_argument("--end", type=date.fromisoformat, required=True)
    rep.add_argument("--rules", type=Path)
    bal = sub.add_parser("balances", help="account balances")
    bal.add_argument("csv", type=Path)
    bal.add_argument("--rules", type=Path)
    return parser.parse_args(argv)


def load(path: Path, account: str, rules_path: Path | None, currency: str = "USD") -> Journal:
    transactions = parse_csv(path.read_text(), Dialect(currency=currency), account)
    rules = RuleSet.parse(rules_path.read_text()) if rules_path else RuleSet([])
    journal = Journal()
    for transaction in transactions:
        target = rules.categorize(transaction.description, default="expenses:uncategorized")
        journal.transfer(transaction.posted, transaction.description, account, target, -transaction.amount,
                         reference=transaction.reference)
    return journal


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    journal = load(args.csv, getattr(args, "account", "assets:checking"), args.rules)
    if args.command == "report":
        print(render(monthly(journal, args.start, args.end)))
    elif args.command == "balances":
        for name, amount in journal.balances().items():
            print(f"{name:<32}{amount.format():>14}")
    else:
        for entry in journal.entries:
            for posting in entry.postings:
                print(f"{entry.posted} {posting.account:<30} {posting.amount.format():>12}  {entry.narration}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
