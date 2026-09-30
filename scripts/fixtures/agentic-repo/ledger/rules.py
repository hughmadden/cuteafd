"""Categorization rules: ``pattern => account`` lines, first match wins.

A pattern is a case-insensitive regular expression searched in the transaction
description. Lines starting with ``#`` and blank lines are ignored. A rule may carry
tags after the account: ``coffee|espresso => expenses:food:dining #treat``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


class RuleError(ValueError):
    pass


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern
    account: str
    tags: tuple[str, ...] = ()
    line: int = 0

    def matches(self, description: str) -> bool:
        return self.pattern.search(description) is not None


@dataclass
class RuleSet:
    rules: list[Rule] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> "RuleSet":
        rules = []
        for number, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=>" not in line:
                raise RuleError(f"line {number}: expected 'pattern => account'")
            pattern, target = (part.strip() for part in line.split("=>", 1))
            words = target.split()
            if not words:
                raise RuleError(f"line {number}: missing account")
            account, tags = words[0], tuple(word[1:] for word in words[1:] if word.startswith("#"))
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error as error:
                raise RuleError(f"line {number}: bad pattern {pattern!r}: {error}") from error
            rules.append(Rule(compiled, account, tags, number))
        return cls(rules)

    def first(self, description: str) -> Rule | None:
        for rule in self.rules:
            if rule.matches(description):
                return rule
        return None

    def categorize(self, description: str, default: str = "expenses:uncategorized") -> str:
        rule = self.first(description)
        return rule.account if rule else default

    def tags(self, description: str) -> tuple[str, ...]:
        rule = self.first(description)
        return rule.tags if rule else ()

    def unused(self, descriptions: list[str]) -> list[Rule]:
        """Rules that match none of ``descriptions`` (candidates for cleanup)."""
        used = {id(self.first(text)) for text in descriptions}
        return [rule for rule in self.rules if id(rule) not in used]

    def render(self) -> str:
        lines = []
        for rule in self.rules:
            tags = "".join(f" #{tag}" for tag in rule.tags)
            lines.append(f"{rule.pattern.pattern} => {rule.account}{tags}")
        return "\n".join(lines)
