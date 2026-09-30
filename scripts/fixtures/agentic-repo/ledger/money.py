"""Money as integer cents with an ISO currency code.

Amounts never touch binary floating point once parsed: arithmetic is on integer
cents, and conversions between currencies round half to even at the cent, the rule
most banks and the ledger's auditors use. ``Money`` is immutable and hashable.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

CURRENCY_DIGITS = {"USD": 2, "EUR": 2, "GBP": 2, "JPY": 0, "CHF": 2, "CAD": 2, "AUD": 2}


class MoneyError(ValueError):
    """An amount could not be parsed or two amounts do not share a currency."""


def round_half_even(numerator: int, denominator: int) -> int:
    """Divide two integers and round the quotient half to even.

    ``round_half_even(5, 2) == 2`` and ``round_half_even(7, 2) == 4``; the sign of the
    result follows the sign of the exact quotient, and ties go to the even neighbour on
    both sides of zero, so ``round_half_even(-5, 2) == -2``.
    """
    if denominator == 0:
        raise ZeroDivisionError("round_half_even by zero")
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    quotient, remainder = divmod(abs(numerator), denominator)
    twice = 2 * remainder
    if twice > denominator or (twice == denominator and quotient % 2 == 1):
        quotient += 1
    if numerator < 0:
        return -quotient - (1 if remainder else 0)
    return quotient


@dataclass(frozen=True, order=True)
class Money:
    """An amount of ``cents`` in ``currency`` (minor units for zero-digit currencies)."""

    cents: int
    currency: str = "USD"

    @classmethod
    def parse(cls, text: str, currency: str = "USD") -> "Money":
        """Parse ``"1,234.56"``, ``"-3.1"``, ``"(12.00)"`` (accounting negative) or ``"7"``."""
        raw = text.strip().replace(",", "")
        negative = raw.startswith("(") and raw.endswith(")")
        if negative:
            raw = raw[1:-1]
        try:
            value = Decimal(raw)
        except InvalidOperation as error:
            raise MoneyError(f"not an amount: {text!r}") from error
        digits = CURRENCY_DIGITS.get(currency, 2)
        scaled = value.scaleb(digits)
        if scaled != scaled.to_integral_value():
            raise MoneyError(f"{text!r} has more than {digits} decimal digits for {currency}")
        cents = int(scaled)
        return cls(-cents if negative else cents, currency)

    @classmethod
    def zero(cls, currency: str = "USD") -> "Money":
        return cls(0, currency)

    def _check(self, other: "Money") -> None:
        if not isinstance(other, Money):
            raise TypeError(f"expected Money, got {type(other).__name__}")
        if other.currency != self.currency:
            raise MoneyError(f"currency mismatch: {self.currency} vs {other.currency}")

    def __add__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.cents + other.cents, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.cents - other.cents, self.currency)

    def __neg__(self) -> "Money":
        return Money(-self.cents, self.currency)

    def __abs__(self) -> "Money":
        return Money(abs(self.cents), self.currency)

    def __bool__(self) -> bool:
        return self.cents != 0

    def scale(self, numerator: int, denominator: int) -> "Money":
        """Multiply by an exact fraction, rounding half to even at the minor unit."""
        return Money(round_half_even(self.cents * numerator, denominator), self.currency)

    def split(self, parts: int) -> list["Money"]:
        """Split into ``parts`` amounts that differ by at most one minor unit and sum exactly."""
        if parts <= 0:
            raise ValueError("parts must be positive")
        base, extra = divmod(self.cents, parts)
        return [Money(base + (1 if i < extra else 0), self.currency) for i in range(parts)]

    def convert(self, rate_numerator: int, rate_denominator: int, currency: str) -> "Money":
        """Convert at ``rate_numerator / rate_denominator`` units of ``currency`` per unit."""
        source = CURRENCY_DIGITS.get(self.currency, 2)
        target = CURRENCY_DIGITS.get(currency, 2)
        numerator = self.cents * rate_numerator * 10 ** target
        denominator = rate_denominator * 10 ** source
        return Money(round_half_even(numerator, denominator), currency)

    def format(self, symbol: bool = True) -> str:
        digits = CURRENCY_DIGITS.get(self.currency, 2)
        sign = "-" if self.cents < 0 else ""
        whole, part = divmod(abs(self.cents), 10 ** digits) if digits else (abs(self.cents), 0)
        body = f"{whole:,}" + (f".{part:0{digits}d}" if digits else "")
        prefix = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}.get(self.currency, "") if symbol else ""
        suffix = "" if prefix or not symbol else f" {self.currency}"
        return f"{sign}{prefix}{body}{suffix}"

    def __str__(self) -> str:
        return self.format()


def total(amounts, currency: str = "USD") -> Money:
    """Sum ``amounts`` (an iterable of Money) starting from zero in ``currency``."""
    result = Money.zero(currency)
    for amount in amounts:
        result = result + amount
    return result
