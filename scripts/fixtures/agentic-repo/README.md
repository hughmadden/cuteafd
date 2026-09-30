# ledger

A small double-entry bookkeeping library for personal finances: import bank CSV exports,
categorize them with rules, post balanced entries and print monthly reports.

```
python -m ledger.cli import export.csv --rules rules.txt
python -m ledger.cli report export.csv --start 2024-01-01 --end 2024-12-31 --rules rules.txt
```

## Design

- Amounts are integer cents (`ledger.money.Money`); rounding is half to even at the cent.
- Accounts are colon-separated paths under five roots: assets, liabilities, equity, income,
  expenses (`ledger.accounts`).
- A `Journal` holds balanced `Entry` objects; balances include descendants.
- Importers (`ledger.importers`) map one institution's export onto `ledger.parser.Dialect`.
- Exchange rates come from a provider behind `ledger.rates.RateCache` (time-bounded).
- Reports (`ledger.report`) group expenses by category depth per calendar month.

## Tests

Tests are plain functions in `tests/` using bare `assert`; run them with the repository's
test runner (the `run_tests` tool in the agent harness, or `python -m pytest` locally).
