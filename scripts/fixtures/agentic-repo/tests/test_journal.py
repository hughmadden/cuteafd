from datetime import date

from ledger.journal import Entry, Journal, Posting, UnbalancedEntry
from ledger.money import Money


def test_transfer_balances():
    j = Journal()
    j.transfer(date(2024, 1, 2), "Opening", "equity:opening", "assets:checking", Money(100000))
    j.transfer(date(2024, 1, 3), "Rent", "assets:checking", "expenses:rent", Money(80000))
    assert j.balance("assets:checking") == Money(20000)
    assert j.balance("expenses") == Money(80000)
    assert not j.trial_balance()


def test_unbalanced_entry_rejected():
    try:
        Journal().post(Entry(date(2024, 1, 1), "bad", [Posting("assets:cash", Money(5)),
                                                      Posting("expenses:fees", Money(-4))]))
    except UnbalancedEntry:
        pass
    else:
        raise AssertionError("unbalanced entry posted")
