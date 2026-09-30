"""Sales-tax helpers: inclusive and exclusive amounts at basis-point rates."""
from __future__ import annotations

from ledger.money import Money

BASIS = 10_000


def add_tax(net: Money, rate_bp: int) -> tuple[Money, Money]:
    """(tax, gross) for a net amount at ``rate_bp`` basis points."""
    tax = net.scale(rate_bp, BASIS)
    return tax, net + tax


def extract_tax(gross: Money, rate_bp: int) -> tuple[Money, Money]:
    """(tax, net) contained in a tax-inclusive amount."""
    net = gross.scale(BASIS, BASIS + rate_bp)
    return gross - net, net


def blended_rate(lines: list[tuple[Money, int]]) -> int:
    """Weighted basis-point rate over ``(net, rate_bp)`` lines (0 for no net amount)."""
    net = sum(amount.cents for amount, _ in lines)
    if not net:
        return 0
    return round(sum(amount.cents * rate for amount, rate in lines) / net)
