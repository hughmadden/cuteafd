from datetime import date

from ledger.budget import Budget
from ledger.journal import Journal
from ledger.money import Money


def test_rollover_carries_unspent_allowance():
    j = Journal()
    j.transfer(date(2024, 3, 5), "Groceries", "assets:checking", "expenses:food:groceries", Money(20000))
    j.transfer(date(2024, 4, 5), "Groceries", "assets:checking", "expenses:food:groceries", Money(45000))
    budget = Budget()
    budget.add("expenses:food", Money(30000))
    status = budget.status(j, [date(2024, 3, 1), date(2024, 4, 1)])[0]
    assert status.available == Money(40000)
    assert status.spent == Money(45000)
    assert status.overspent
