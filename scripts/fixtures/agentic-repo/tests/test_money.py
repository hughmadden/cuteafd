from ledger.money import Money, MoneyError, round_half_even, total


def test_parse_forms():
    assert Money.parse("1,234.56").cents == 123456
    assert Money.parse("(12.00)").cents == -1200
    assert Money.parse("7", "JPY") == Money(7, "JPY")
    try:
        Money.parse("1.234")
    except MoneyError:
        pass
    else:
        raise AssertionError("three decimals accepted for USD")


def test_arithmetic_and_currency_mismatch():
    assert Money(150) + Money(50) == Money(200)
    assert -Money(5) == Money(-5)
    try:
        Money(1, "USD") + Money(1, "EUR")
    except MoneyError:
        pass
    else:
        raise AssertionError("mixed currencies added")


def test_positive_rounding():
    assert round_half_even(5, 2) == 2
    assert round_half_even(7, 2) == 4
    assert round_half_even(10, 4) == 2
    assert Money(1005).scale(1, 2) == Money(502)


def test_negative_rounding():
    assert round_half_even(-5, 2) == -2
    assert round_half_even(-7, 2) == -4
    assert round_half_even(-10, 3) == -3
    assert round_half_even(-6, 2) == -3
    assert Money(-1005).scale(1, 2) == Money(-502)


def test_split_and_total():
    parts = Money(1000).split(3)
    assert [p.cents for p in parts] == [334, 333, 333]
    assert total(parts) == Money(1000)


def test_convert_and_format():
    assert Money(10000, "USD").convert(9, 10, "EUR") == Money(9000, "EUR")
    assert Money(123456).format() == "$1,234.56"
    assert Money(-5, "EUR").format() == "-€0.05"
    assert Money(1500, "JPY").format() == "¥1,500"
