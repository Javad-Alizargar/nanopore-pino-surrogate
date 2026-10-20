#!/usr/bin/env python3
"""Normalize number/unit hygiene in the canonical markdown sources.

Fixes the ASCII forms that the previous DOCX step then mangled downstream:

  * bare scientific notation in prose (``9.936e-10``) ->
    ``$9.936\\times10^{-10}$`` so it renders as native math, not ASCII ``e``;
  * ASCII caret units (``mol m^-3``) -> ``mol m$^{-3}$``.

The transform is idempotent: existing math, code spans, URLs, and DOIs are
protected and never edited.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "manuscript" / "Manuscript.md",
           ROOT / "manuscript" / "Supplementary_Materials.md"]

PROTECT = [
    re.compile(r"\$\$.*?\$\$", re.S),
    re.compile(r"\$[^$\n]*\$"),
    re.compile(r"```.*?```", re.S),
    re.compile(r"`[^`]*`"),
    re.compile(r"https?://\S+"),
    re.compile(r"doi:\S+", re.I),
    re.compile(r"10\.\d{4,9}/\S+"),
    re.compile(r"arXiv:\d{4}\.\d{4,5}", re.I),
]

SCI = re.compile(r"(?<![\w.$\\])(\d+(?:\.\d+)?)e([+-]?\d+)(?![\w])")
UNIT = re.compile(r"\b(mol m)\^\{?(-?\d)\}?")
# A bare letter followed by a math superscript/subscript (``m$^{-3}$``) renders
# in OMML as an empty-base superscript (a placeholder box); pull the base inside.
ORPHAN = re.compile(r"(?<![\w$\\])([A-Za-z])\$([\^_])([^$\n]*)\$")


def transform(text: str) -> tuple[str, int]:
    n = 0

    # --- pass 0: pull orphan superscript bases into math ----------------------
    spans = [(m.start(), m.end()) for pat in PROTECT for m in pat.finditer(text)]
    spans.sort()

    def protected0(start: int) -> bool:
        return any(s <= start < e for s, e in spans)

    out, cursor = [], 0
    for m in ORPHAN.finditer(text):
        out.append(text[cursor:m.start()])
        if protected0(m.start()):
            out.append(m.group(0))
        else:
            out.append(rf"${m.group(1)}{m.group(2)}{m.group(3)}$")
            n += 1
        cursor = m.end()
    out.append(text[cursor:])
    text = "".join(out)

    # --- pass 1: caret units -------------------------------------------------
    spans = [(m.start(), m.end()) for pat in PROTECT for m in pat.finditer(text)]
    spans.sort()

    def protected(start: int) -> bool:
        return any(s <= start < e for s, e in spans)

    out, cursor = [], 0
    for m in UNIT.finditer(text):
        out.append(text[cursor:m.start()])
        if protected(m.start()):
            out.append(m.group(0))
        else:
            out.append(rf"{m.group(1)}$^{{{m.group(2)}}}$")
            n += 1
        cursor = m.end()
    out.append(text[cursor:])
    text = "".join(out)

    # --- pass 2: bare scientific notation (offsets refreshed) ----------------
    spans = [(m.start(), m.end()) for pat in PROTECT for m in pat.finditer(text)]
    spans.sort()

    def protected2(start: int) -> bool:
        return any(s <= start < e for s, e in spans)

    out, cursor = [], 0
    for m in SCI.finditer(text):
        out.append(text[cursor:m.start()])
        if protected2(m.start()):
            out.append(m.group(0))
        else:
            mant, exp = m.group(1), m.group(2)
            sign = "-" if exp.startswith("-") else ""
            digits = exp.lstrip("+-")
            out.append(rf"${mant}\times10^{{{sign}{digits}}}$")
            n += 1
        cursor = m.end()
    out.append(text[cursor:])
    return "".join(out), n


def main() -> None:
    for path in TARGETS:
        src = path.read_text(encoding="utf-8")
        fixed, n = transform(src)
        if fixed != src:
            path.write_text(fixed, encoding="utf-8")
        print(f"{path.name}: {n} normalizations; changed={fixed != src}")


if __name__ == "__main__":
    main()
