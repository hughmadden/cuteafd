"""Bank-specific importers built on ``ledger.parser``.

Each importer knows one institution's export: its column names, date format, sign
convention and quirks (a trailer line, a currency column). ``detect`` picks an importer
from a header line.
"""
from ledger.importers.bank import FirstRiverImporter
from ledger.importers.card import NorthCardImporter
from ledger.importers.paypal import PaypalImporter

IMPORTERS = (FirstRiverImporter, NorthCardImporter, PaypalImporter)


def detect(header: str):
    for importer in IMPORTERS:
        if importer.accepts(header):
            return importer()
    return None
