from datetime import date

from ledger.dates import add_months, fiscal_year, month_end, month_start, months_between, quarter


def test_month_start_and_quarter():
    assert month_start(date(2024, 5, 17)) == date(2024, 5, 1)
    assert quarter(date(2024, 5, 17)) == 2
    assert fiscal_year(date(2024, 10, 1), first_month=10) == 2025


def test_month_end_regular_months():
    assert month_end(date(2024, 2, 10)) == date(2024, 2, 29)
    assert month_end(date(2023, 4, 1)) == date(2023, 4, 30)


def test_december_month_end():
    assert month_end(date(2024, 12, 15)) == date(2024, 12, 31)


def test_add_months_clamps():
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2024, 11, 30), 3) == date(2025, 2, 28)


def test_months_between():
    assert months_between(date(2024, 11, 5), date(2025, 1, 2)) == [date(2024, 11, 1), date(2024, 12, 1),
                                                                    date(2025, 1, 1)]
    assert months_between(date(2024, 2, 1), date(2024, 1, 1)) == []
