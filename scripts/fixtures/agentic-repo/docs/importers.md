# Importers

| Institution | Class | Date format | Sign | Notes |
| --- | --- | --- | --- | --- |
| First River Bank | `FirstRiverImporter` | ISO | signed | drops the `Total` trailer line |
| North Card | `NorthCardImporter` | `%m/%d/%Y` | charges positive | flips the sign |
| PayPal | `PaypalImporter` | `%d/%m/%Y` | net column | keeps only `Completed` rows, per-row currency |

`ledger.importers.detect(header_line)` returns the first importer whose `accepts` matches.
