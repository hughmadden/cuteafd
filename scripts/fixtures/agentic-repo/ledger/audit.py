"""Consistency checks over a journal: balance, dates, duplicates, closed accounts."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ledger.journal import Journal


@dataclass(frozen=True)
class Finding:
    severity: str
    message: str
    entry: int | None = None


def audit(journal: Journal) -> list[Finding]:
    findings: list[Finding] = []
    if journal.trial_balance():
        findings.append(Finding("error", f"trial balance is {journal.trial_balance()}"))
    previous = None
    for index, entry in enumerate(journal.entries):
        if previous is not None and entry.posted < previous:
            findings.append(Finding("warning", f"entry posted out of order ({entry.posted})", index))
        previous = entry.posted
        for posting in entry.postings:
            if posting.account in journal.chart and journal.chart.get(posting.account).closed:
                findings.append(Finding("error", f"posting to closed account {posting.account}", index))
    references = Counter(entry.reference for entry in journal.entries if entry.reference)
    for reference, count in sorted(references.items()):
        if count > 1:
            findings.append(Finding("warning", f"reference {reference} appears {count} times"))
    return findings


def summarize(findings: list[Finding]) -> str:
    counts = Counter(finding.severity for finding in findings)
    return ", ".join(f"{counts[level]} {level}(s)" for level in ("error", "warning") if counts[level]) or "clean"
