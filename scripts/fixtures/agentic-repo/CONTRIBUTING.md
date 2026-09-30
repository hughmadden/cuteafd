# Contributing

- Keep amounts in integer cents; never convert through `float`.
- Every bug fix comes with a test in `tests/` that fails before the fix.
- Keep functions small and typed; prefer dataclasses for records.
- Run the whole test suite before sending a change; a change that fixes one test must not
  break another.
- Importers must not change `ledger.parser` behaviour for other institutions.
