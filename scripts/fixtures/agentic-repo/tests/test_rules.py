from ledger.rules import RuleError, RuleSet

RULES = """
# groceries first
market|grocer => expenses:food:groceries
coffee|espresso => expenses:food:dining #treat
rent => expenses:rent
"""


def test_first_match_wins_and_tags():
    rules = RuleSet.parse(RULES)
    assert rules.categorize("Corner Market") == "expenses:food:groceries"
    assert rules.categorize("ESPRESSO BAR") == "expenses:food:dining"
    assert rules.tags("espresso bar") == ("treat",)
    assert rules.categorize("unknown") == "expenses:uncategorized"


def test_bad_rule_lines():
    for text in ("no arrow here", "( => expenses:x"):
        try:
            RuleSet.parse(text)
        except RuleError:
            continue
        raise AssertionError(f"{text!r} accepted")
