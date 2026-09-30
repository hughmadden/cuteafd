"""Render a monthly report as a standalone HTML table."""
from __future__ import annotations

from html import escape

from ledger.report import MonthRow

STYLE = "body{font-family:sans-serif}td.num{text-align:right;font-variant-numeric:tabular-nums}"


def table(rows: list[MonthRow], title: str = "Monthly report") -> str:
    body = []
    for row in rows:
        body.append(f"<tr><th>{row.start:%Y-%m}</th><td class=num>{escape(row.income.format())}</td>"
                    f"<td class=num>{escape(row.expenses.format())}</td><td class=num>{escape(row.net.format())}</td></tr>")
        for category, amount in row.by_category.items():
            body.append(f"<tr class=category><td>{escape(category)}</td><td></td>"
                        f"<td class=num>{escape(amount.format())}</td><td></td></tr>")
    return (f"<!doctype html><title>{escape(title)}</title><style>{STYLE}</style><h1>{escape(title)}</h1>"
            "<table><tr><th>Month</th><th>Income</th><th>Expenses</th><th>Net</th></tr>" + "".join(body) + "</table>")
