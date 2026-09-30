from datetime import date

from ledger.journal import Journal
from ledger.money import Money
from ledger.report import monthly, render


def journal():
    j = Journal()
    j.transfer(date(2024, 11, 1), "Salary", "income:salary", "assets:checking", Money(300000))
    j.transfer(date(2024, 11, 3), "Groceries", "assets:checking", "expenses:food:groceries", Money(8450))
    j.transfer(date(2024, 12, 20), "Rent", "assets:checking", "expenses:rent", Money(150000))
    j.transfer(date(2024, 12, 31), "Dinner", "assets:checking", "expenses:food:dining", Money(6200))
    return j


def test_november_report():
    row = monthly(journal(), date(2024, 11, 1), date(2024, 11, 30))[0]
    assert row.income == Money(300000)
    assert row.expenses == Money(8450)
    assert row.by_category == {"expenses:food": Money(8450)}


def test_december_report():
    row = monthly(journal(), date(2024, 12, 1), date(2024, 12, 31))[0]
    assert row.end == date(2024, 12, 31)
    assert row.expenses == Money(156200)
    assert "2024-12" in render([row])
