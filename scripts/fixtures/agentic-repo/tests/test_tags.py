from ledger.tags import extract, histogram, rename, strip


def test_extract_and_strip():
    assert extract("Flight #Travel to #q3-offsite #travel") == ("travel", "q3-offsite")
    assert extract("issue#12 is not a tag") == ()
    assert strip("Dinner #work  with team") == "Dinner with team"


def test_histogram_and_rename():
    assert histogram(["#a #b", "#a"])["a"] == 2
    assert rename("Taxi #trip #trip-2", "trip", "travel") == "Taxi #travel #trip-2"
