# Changelog

## 0.9.3
- PayPal importer skips pending and reversed activity.
- `RateCache` takes an injectable clock; tests no longer sleep.
- `Money.convert` handles zero-digit currencies on either side.

## 0.9.2
- Envelope budgets with rollover (`ledger.budget`).
- HTML export of monthly reports.

## 0.9.1
- Reconciliation of imported transactions against the journal within a date window.
- Categorization rules may carry tags (`=> expenses:food:dining #treat`).

## 0.9.0
- SQLite storage alongside JSON.
- Accounting-style negatives `(12.00)` in `Money.parse`.
