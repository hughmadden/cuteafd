# Architecture

```
importers/*  ->  parser (Dialect, split_csv_line, parse_csv)  ->  Transaction
rules        ->  categorize(description)                       ->  account name
journal      ->  Entry(postings) checked balanced, Chart opens accounts on demand
report       ->  month_row / monthly over journal.between(month_start, month_end)
budget       ->  envelopes over month_row categories, with rollover
rates        ->  RateCache(provider, ttl, clock) in front of StaticRates or a remote provider
storage/*    ->  JSON and SQLite round trips
export/*     ->  CSV (quoting compatible with the parser) and HTML tables
audit        ->  trial balance, ordering, duplicate references, closed accounts
```

## Money

`Money(cents, currency)`; `CURRENCY_DIGITS` gives minor-unit digits per currency. All
scaling and conversion goes through `round_half_even(numerator, denominator)`, whose result
must be symmetric around zero: `round_half_even(-n, d) == -round_half_even(n, d)`.

## CSV

Fields may be quoted with `"`; a quoted field may contain commas and doubled quotes. The
exporter in `export/csv_export.py` writes exactly what `split_csv_line` must read back.

## Dates

Reporting periods are calendar months: `month_start(d)` .. `month_end(d)` inclusive, for
every month including December.

## Rates

`RateCache.get` serves a cached rate while it is younger than `ttl` seconds of the injected
clock and refetches it afterwards.
