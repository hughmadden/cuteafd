"""The chart of accounts.

Accounts form a tree named with colons (``expenses:food:groceries``). Each account has a
kind that fixes the sign convention of its balance: assets and expenses are debit-normal,
liabilities, equity and income are credit-normal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AccountKind(Enum):
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    INCOME = "income"
    EXPENSE = "expense"

    @property
    def debit_normal(self) -> bool:
        return self in (AccountKind.ASSET, AccountKind.EXPENSE)

    @classmethod
    def from_root(cls, root: str) -> "AccountKind":
        roots = {"assets": cls.ASSET, "liabilities": cls.LIABILITY, "equity": cls.EQUITY,
                 "income": cls.INCOME, "expenses": cls.EXPENSE}
        try:
            return roots[root]
        except KeyError:
            raise ValueError(f"unknown account root {root!r}; expected one of {sorted(roots)}") from None


class AccountError(ValueError):
    """An account name is malformed or an account is missing from the chart."""


@dataclass
class Account:
    name: str
    kind: AccountKind
    currency: str = "USD"
    closed: bool = False
    note: str = ""
    children: list["Account"] = field(default_factory=list, repr=False)

    @property
    def parent_name(self) -> str | None:
        return self.name.rsplit(":", 1)[0] if ":" in self.name else None

    @property
    def leaf(self) -> str:
        return self.name.rsplit(":", 1)[-1]

    @property
    def depth(self) -> int:
        return self.name.count(":")

    def walk(self):
        yield self
        for child in sorted(self.children, key=lambda account: account.name):
            yield from child.walk()


def validate_name(name: str) -> None:
    parts = name.split(":")
    if not all(parts):
        raise AccountError(f"empty component in account name {name!r}")
    for part in parts:
        if not all(ch.isalnum() or ch in "-_" for ch in part):
            raise AccountError(f"invalid character in account component {part!r}")
    AccountKind.from_root(parts[0])


class Chart:
    """All accounts, created on demand with their parents."""

    def __init__(self, currency: str = "USD"):
        self.currency = currency
        self._accounts: dict[str, Account] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._accounts

    def __len__(self) -> int:
        return len(self._accounts)

    def get(self, name: str) -> Account:
        try:
            return self._accounts[name]
        except KeyError:
            raise AccountError(f"no account {name!r}") from None

    def open(self, name: str, currency: str | None = None, note: str = "") -> Account:
        """Open ``name`` and any missing parents; returns the (possibly existing) account."""
        validate_name(name)
        if name in self._accounts:
            account = self._accounts[name]
            if account.closed:
                raise AccountError(f"account {name!r} is closed")
            return account
        kind = AccountKind.from_root(name.split(":", 1)[0])
        account = Account(name, kind, currency or self.currency, note=note)
        parent = account.parent_name
        if parent is not None:
            self.open(parent, currency).children.append(account)
        self._accounts[name] = account
        return account

    def close(self, name: str) -> None:
        account = self.get(name)
        for descendant in account.walk():
            descendant.closed = True

    def roots(self) -> list[Account]:
        return sorted((a for a in self._accounts.values() if a.parent_name is None), key=lambda a: a.name)

    def names(self, prefix: str = "") -> list[str]:
        return sorted(name for name in self._accounts if name.startswith(prefix))

    @classmethod
    def standard(cls, currency: str = "USD") -> "Chart":
        chart = cls(currency)
        for name in ("assets:checking", "assets:savings", "assets:cash", "liabilities:card",
                     "equity:opening", "income:salary", "income:interest", "expenses:food:groceries",
                     "expenses:food:dining", "expenses:rent", "expenses:utilities", "expenses:transport",
                     "expenses:fees", "expenses:uncategorized"):
            chart.open(name)
        return chart
