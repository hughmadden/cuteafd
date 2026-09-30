from ledger.rates import RateCache, RateError, StaticRates


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_static_rates_and_inverse():
    rates = StaticRates({("USD", "EUR"): (9, 10)})
    assert rates.rate("USD", "EUR") == (9, 10)
    assert rates.rate("EUR", "USD") == (10, 9)
    assert rates.rate("GBP", "GBP") == (1, 1)
    try:
        rates.rate("USD", "JPY")
    except RateError:
        pass
    else:
        raise AssertionError("unknown pair answered")


def test_expired_rate_refetched():
    provider = StaticRates({("USD", "EUR"): (9, 10)})
    clock = Clock()
    cache = RateCache(provider, ttl=10, clock=clock)
    assert cache.get("USD", "EUR") == (9, 10)
    clock.now = 5
    assert cache.get("USD", "EUR") == (9, 10)
    assert (provider.calls, cache.hits) == (1, 1)
    clock.now = 11
    provider.table[("USD", "EUR")] = (19, 20)
    assert cache.get("USD", "EUR") == (19, 20)
    assert (provider.calls, cache.misses) == (2, 2)


def test_invalidate():
    provider = StaticRates({("USD", "EUR"): (9, 10), ("GBP", "EUR"): (6, 5)})
    cache = RateCache(provider, ttl=10, clock=Clock())
    cache.get("USD", "EUR")
    cache.get("GBP", "EUR")
    assert cache.invalidate("USD") == 1
    assert cache.invalidate() == 1
