#!/usr/bin/env python3
"""Apply scripts/build/rename-map.tsv to Rust sources.

  rename_paths.py [--map FILE] [--check] PATH...   rewrite files in place (--check: list, don't write)

rewrite(path, text) is imported by rebase-across-move.sh for hunk lines of
carried branches, so the pass and the carried branches use one rule set.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_MAP = Path(__file__).resolve().with_name("rename-map.tsv")
_GROUP = re.compile(r"\buse crate::\{")


def load(map_path: Path = DEFAULT_MAP) -> list[tuple[str, re.Pattern[str], str]]:
    rules = []
    for line in map_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        scope, rx, rep = line.split("\t")
        rules.append((scope, re.compile(rx), rep))
    return rules


def _split_items(body: str) -> list[str]:
    items, depth, cur = [], 0, []
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        items.append("".join(cur).strip())
    return items


def _split_groups(text: str, touched) -> str:
    """use crate::{a::X, b::{Y, Z}}; -> one use per item, when any item is renamed."""
    out, pos = [], 0
    for m in _GROUP.finditer(text):
        start = m.end()
        depth, i = 1, start
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth or not text[i:].lstrip().startswith(";"):
            continue
        end = text.index(";", i) + 1
        items = _split_items(text[start:i - 1])
        if not any(touched("crate::" + item) for item in items):
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        indent = re.match(r"[ \t]*", text[line_start:m.start()]).group(0)
        prefix = text[line_start:m.start()][len(indent):]  # e.g. "pub(crate) "
        joined = ("\n" + indent).join(f"{prefix}use crate::{item};" for item in items)
        out.append(text[pos:line_start] + indent + joined)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def rewrite(path: str, text: str, rules=None) -> str:
    rules = rules if rules is not None else load()
    active = [(rx, rep) for scope, rx, rep in rules if path.startswith(scope)]
    if not active:
        return text
    text = _split_groups(text, lambda s: any(rx.search(s) for rx, _ in active))
    for rx, rep in active:
        text = rx.sub(rep, text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    rules = load(args.map)
    changed = 0
    for root in args.paths:
        for file in sorted(root.rglob("*.rs")) if root.is_dir() else [root]:
            rel = file.as_posix()
            text = file.read_text(encoding="utf-8")
            new = rewrite(rel, text, rules)
            if new != text:
                changed += 1
                if args.check:
                    print(rel)
                else:
                    file.write_text(new, encoding="utf-8")
    print(f"{changed} file(s) {'would change' if args.check else 'rewritten'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
