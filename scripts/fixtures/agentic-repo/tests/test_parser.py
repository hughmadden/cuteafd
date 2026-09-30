from datetime import date

from ledger.money import Money
from ledger.parser import Dialect, ParseError, parse_csv, split_csv_line

EXPORT = """Date,Description,Amount,Reference
2024-03-01,Coffee Roasters,-4.50,A1
2024-03-02,Payroll,2500.00,A2
"""


def test_plain_fields():
    assert split_csv_line("a,b,c\r\n") == ["a", "b", "c"]


def test_parse_export():
    rows = parse_csv(EXPORT)
    assert [row.amount for row in rows] == [Money(-450), Money(250000)]
    assert rows[0].posted == date(2024, 3, 1)
    assert rows[1].reference == "A2"


def test_missing_column():
    try:
        parse_csv("Date,Amount\n2024-01-01,1.00\n")
    except ParseError as error:
        assert "Description" in str(error)
    else:
        raise AssertionError("missing column accepted")


def test_quoted_commas():
    assert split_csv_line('a,"b, c",d') == ["a", "b, c", "d"]
    rows = parse_csv('Date,Description,Amount,Reference\n2024-03-03,"Smith, Jones & Co",-120.00,A3\n')
    assert rows[0].description == "Smith, Jones & Co"
    assert rows[0].amount == Money(-12000)


def test_doubled_quotes():
    assert split_csv_line('"say ""hi""",x') == ['say "hi"', "x"]


def test_dialect_aliases():
    dialect = Dialect(amount="Value", aliases={"Value": "Betrag"})
    rows = parse_csv("Date,Description,Betrag\n2024-01-05,Bakery,-3.20\n", dialect)
    assert rows[0].amount == Money(-320)
