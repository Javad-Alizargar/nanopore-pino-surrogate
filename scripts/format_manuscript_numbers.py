#!/usr/bin/env python3
"""Round long decimal values in narrative markdown to 3-4 significant figures.

Exact values remain in results/statistics.json and results/tables/*.csv.
Protected regions: code spans, LaTeX math, URLs, and the reference list.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "manuscript/Manuscript.md", ROOT / "manuscript/Supplementary_Materials.md"]
MATH = [
    re.compile(r"\$\$.*?\$\$", re.S),
    re.compile(r"\$[^$\n]*\$"),
]

PROTECT = [
    re.compile(r"```.*?```", re.S),
    re.compile(r"`[^`]*`"),
    re.compile(r"https?://\S+"),
    re.compile(r"doi:\S+", re.I),
    re.compile(r"10\.\d{4,9}/\S+"),
    re.compile(r"arXiv:\d{4}\.\d{4,5}", re.I),
]
NUM = re.compile(r"(?<![\w.])-?\d+\.\d+(?:[eE][+-]?\d+)?(?!\w)")
IGNORE = {"298.15"}


def sig_figs(text: str) -> int:
    mant = text.lower().split("e")[0].lstrip("-").replace(".", "").lstrip("0")
    return len(mant)


def fmt(x: float) -> str:
    if x == 0:
        return "0"
    exp = math.floor(math.log10(abs(x)))
    if -4 <= exp < 6:
        dec = max(0, 4 - exp - 1)
        out = f"{x:.{dec}f}".rstrip("0").rstrip(".")
        return out if out not in ("", "-0") else "0"
    out = f"{x:.3e}"
    return re.sub(r"e([+-])0+(\d+)", r"e\1\2", out)


def round_text(text: str, changes: list[dict]) -> str:
    hard: list[tuple[int, int]] = []
    for pat in PROTECT:
        hard.extend((m.start(), m.end()) for m in pat.finditer(text))
    ref_head = text.find("## References")
    if ref_head >= 0:
        hard.append((ref_head, len(text)))
    math_spans = [(m.start(), m.end(), m.group(0)) for pat in MATH for m in pat.finditer(text)]

    def protected(start: int, end: int) -> bool:
        if any(a <= start and end <= b for a, b in hard):
            return True
        for a, b, txt in math_spans:
            if a <= start and end <= b:
                return "\\" in txt
        return False

    out, last = [], 0
    for m in NUM.finditer(text):
        old = m.group(0)
        if protected(m.start(), m.end()) or old in IGNORE or sig_figs(old) <= 4:
            continue
        try:
            new = fmt(float(old))
        except ValueError:
            continue
        if new == old:
            continue
        changes.append({"old": old, "new": new})
        out.append(text[last:m.start()])
        out.append(new)
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report without writing")
    args = ap.parse_args()
    all_changes = {}
    for path in TARGETS:
        text = path.read_text()
        changes = []
        new = round_text(text, changes)
        all_changes[str(path.relative_to(ROOT))] = changes
        print(f"{path.name}: {len(changes)} replacements")
        if changes:
            for c in changes[:8]:
                print(f"  {c['old']} -> {c['new']}")
        if not args.check:
            path.write_text(new)
    if not args.check:
        out = ROOT / "results" / "number_format_changes.json"
        out.write_text(json.dumps(all_changes, indent=2))
        print(f"log: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
