from ledger.util.cache import LRU
from ledger.util.config import Config
from ledger.util.iterutil import chunked, group_by, unique
from ledger.util.text import payee, slug, truncate


def test_text_helpers():
    assert payee("POS PURCHASE Café Luna CARD 1234") == "cafe luna"
    assert slug("Crème Brûlée & Co.") == "creme-brulee-co"
    assert truncate("abcdef", 4) == "abc…"


def test_iter_helpers():
    assert list(chunked(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert group_by(["aa", "b", "cc"], len) == {2: ["aa", "cc"], 1: ["b"]}
    assert unique([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_lru_and_config():
    cache = LRU(2)
    for key in "abca":
        cache.get(key, lambda: key.upper())
    assert (cache.hits, cache.misses, len(cache)) == (0, 4, 2)
    config = Config.parse('[rates]\nttl = 60  # seconds\nprovider = "static"\n')
    assert config.integer("rates", "ttl") == 60 and config.get("rates", "provider") == "static"
