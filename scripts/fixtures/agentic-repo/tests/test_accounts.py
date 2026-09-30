from ledger.accounts import AccountError, AccountKind, Chart


def test_open_creates_parents():
    chart = Chart()
    chart.open("expenses:food:groceries")
    assert chart.names() == ["expenses", "expenses:food", "expenses:food:groceries"]
    assert chart.get("expenses:food").kind is AccountKind.EXPENSE


def test_invalid_names():
    for name in ("food", "expenses::x", "assets:ch ecking"):
        try:
            Chart().open(name)
        except (AccountError, ValueError):
            continue
        raise AssertionError(f"{name!r} accepted")


def test_close_blocks_reopen():
    chart = Chart.standard()
    chart.close("expenses:food")
    try:
        chart.open("expenses:food:dining")
    except AccountError:
        pass
    else:
        raise AssertionError("closed account reopened")
