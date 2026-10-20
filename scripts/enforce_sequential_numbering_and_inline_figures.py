#!/usr/bin/env python3
"""Enforce strict sequential numbering and inline figure placement (Stage 11).

What it does (deterministic, idempotent):

  1. Vancouver first-appearance reference renumbering.  The reference numbers in
     the main text are reassigned so that the first time a citation appears the
     integer is 1, then 2, ... 60.  Bracketed groups are sorted, de-duplicated,
     and collapsed to an en-dash for runs of three or more.  The reference list,
     ``manuscript/literature_db.json``, the L01--L60 rows of
     ``manuscript/claim_evidence.csv``, and
     ``results/literature_claim_verification.json`` are permuted to match.
  2. Supplementary-table renumbering by first citation in the main text, so the
     labels are strictly increasing S1..S18 (labels only; each table stays in
     its thematic supplement section).  The manuscript and supplement callouts
     are rewritten consistently.
  3. Verification of strict sequential equation callouts (main ``Eq. (1)..(8)``,
     supplement ``Eq. (S1)..(S10)``) and of inline figure placement in the
     Markdown (image + caption immediately after the first-mention paragraph)
     and, if a DOCX exists, in the WordprocessingML body.

Usage:
    .venv/bin/python scripts/enforce_sequential_numbering_and_inline_figures.py
    .venv/bin/python scripts/enforce_sequential_numbering_and_inline_figures.py --check
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manuscript" / "Manuscript.md"
SUP = ROOT / "manuscript" / "Supplementary_Materials.md"
CLAIMS = ROOT / "manuscript" / "claim_evidence.csv"
LITDB = ROOT / "manuscript" / "literature_db.json"
LITVER = ROOT / "results" / "literature_claim_verification.json"
DOCX = ROOT / "manuscript" / "Manuscript_PINO_clean.docx"
REPORT = ROOT / "results" / "stage11_numbering_report.json"

REF_RE = re.compile(r"\[([0-9][0-9,\-\u2013 ]*)\]")
TAB_RE = re.compile(
    r"((?:Supplementary\s+)?Tables?\s+)(S\d+(?:\s*(?:,|and|--|\u2013|-)\s*S?\d+)*)"
)
EQ_MAIN_RE = re.compile(r"Eq\.\s*\((\d+)\)")
EQ_SUP_RE = re.compile(r"Eq\.\s*\(S(\d+)\)")
FIG_MENTION = re.compile(r"Figure\s+(\d)")
FIG_IMAGE = re.compile(r"^!\[[^\]]*\]\(([^)]+)\)\s*$")
FIG_CAPTION = re.compile(r"^\*\*Figure\s+(\d+)\.\*\*")
REF_LINE = re.compile(r"^(\d+)\.\s+(.*)$")
SUPP_CAP = re.compile(r"^\*\*Supplementary Table S(\d+)\.\*\*")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def mask_math(text: str) -> str:
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    return re.sub(r"\$[^$]*\$", "", text)


def expand_list(s: str) -> list[int]:
    s = s.replace("\u2013", "-").replace("--", "-").replace("S", "")
    out: list[int] = []
    for part in re.split(r",|;|\band\b", s):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")[0], part.split("-")[-1]
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def fmt_list(nums: list[int], prefix: str = "") -> str:
    nums = sorted(set(nums))
    parts: list[str] = []
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        n = j - i + 1
        if n >= 3:
            parts.append(f"{prefix}{nums[i]}--{prefix}{nums[j]}")
        else:
            parts.extend(f"{prefix}{nums[k]}" for k in range(i, j + 1))
        i = j + 1
    return ", ".join(parts)


def first_appearance(text: str, regex: re.Pattern, kind: str) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for m in regex.finditer(mask_math(text)):
        nums = expand_list(m.group(2) if kind == "table" else m.group(1))
        for n in nums:
            if n not in seen:
                seen.add(n)
                order.append(n)
    return order


def rewrite_refs(text: str, perm: dict[int, int]) -> str:
    def repl(m: re.Match) -> str:
        nums = [perm.get(n, n) for n in expand_list(m.group(1))]
        return f"[{fmt_list(nums)}]"
    return REF_RE.sub(repl, text)


def rewrite_tables(text: str, perm: dict[int, int]) -> str:
    def repl(m: re.Match) -> str:
        nums = [perm.get(n, n) for n in expand_list(m.group(2))]
        return m.group(1) + fmt_list(nums, prefix="S")
    return TAB_RE.sub(repl, text)


def read_doc(path: Path) -> tuple[str, str, object]:
    text = path.read_text()
    if "## References" in text:
        body, refs = text.split("## References", 1)
        return body, "## References" + refs, text
    return text, "", text


# --------------------------------------------------------------------------- #
# stage 1: references
# --------------------------------------------------------------------------- #
def renumber_references() -> dict:
    body, refs, _ = read_doc(MAN)
    order = first_appearance(body, REF_RE, "citation")
    if sorted(order) != list(range(1, 61)):
        raise SystemExit(f"reference first-appearance is not a permutation of 1..60: {sorted(order)}")
    perm = {old: i + 1 for i, old in enumerate(order)}

    old_entries = {}
    for line in refs.splitlines():
        m = REF_LINE.match(line.strip())
        if m:
            old_entries[int(m.group(1))] = m.group(2)
    if sorted(old_entries) != list(range(1, 61)):
        raise SystemExit(f"reference list is not 1..60: {sorted(old_entries)}")

    new_body = rewrite_refs(body, perm)
    new_lines = ["## References", ""]
    for n in range(1, 61):
        new_lines.append(f"{n}. {old_entries[order[n - 1]]}")
    new_refs = "\n".join(new_lines) + "\n"
    new_text = new_body + new_refs

    # literature_db.json
    db = json.loads(LITDB.read_text())
    entries = db["entries"]
    if len(entries) != 60:
        raise SystemExit(f"literature_db has {len(entries)} entries")
    db["entries"] = [entries[order[i] - 1] for i in range(60)]

    # claim_evidence L rows
    with CLAIMS.open(newline="") as fh:
        rows = list(csv.reader(fh))
    header, data = rows[0], rows[1:]
    c_rows = [r for r in data if r and r[0].startswith("C")]
    l_rows = [r for r in data if r and r[0].startswith("L")]
    if len(l_rows) != 60:
        raise SystemExit(f"expected 60 L rows, found {len(l_rows)}")
    new_l = []
    for old in range(1, 61):
        row = l_rows[old - 1]
        row = list(row)
        row[1] = rewrite_refs(row[1], perm)
        m = re.match(r"^\[(\d+)\]", row[4])
        if m:
            row[4] = "[" + str(perm[int(m.group(1))]) + "]" + row[4][m.end():]
        new_l.append((perm[old], row))
    new_l.sort(key=lambda t: t[0])
    for i, (_, row) in enumerate(new_l):
        row[0] = f"L{i + 1:02d}"
    with CLAIMS.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(c_rows)
        w.writerows(row for _, row in new_l)

    # literature_claim_verification.json
    if LITVER.exists():
        ver = json.loads(LITVER.read_text())
        for r in ver.get("rows", []):
            r["ref"] = perm.get(int(r["ref"]), int(r["ref"]))
            if "sentence" in r and r["sentence"]:
                r["sentence"] = rewrite_refs(r["sentence"], perm)
        ver["rows"] = sorted(ver.get("rows", []), key=lambda r: r["ref"])
        LITVER.write_text(json.dumps(ver, indent=1) + "\n")

    MAN.write_text(new_text)
    LITDB.write_text(json.dumps(db, indent=1) + "\n")
    return {"order": order, "perm": {str(k): v for k, v in perm.items()}}


# --------------------------------------------------------------------------- #
# stage 2: supplementary tables
# --------------------------------------------------------------------------- #
def renumber_tables() -> dict:
    body, _, _ = read_doc(MAN)
    order = first_appearance(body, TAB_RE, "table")
    if sorted(order) != list(range(1, 19)):
        raise SystemExit(
            f"supplementary-table first-appearance is not a permutation of 1..18: {sorted(order)}")
    perm = {old: i + 1 for i, old in enumerate(order)}

    man = MAN.read_text()
    sup = SUP.read_text()
    MAN.write_text(rewrite_tables(man, perm))
    SUP.write_text(rewrite_tables(sup, perm))
    return {"order": order, "perm": {str(k): v for k, v in perm.items()}}


# --------------------------------------------------------------------------- #
# stage 3: verification
# --------------------------------------------------------------------------- #
def check_equations(text: str, regex: re.Pattern) -> dict:
    seq = [int(m.group(1)) for m in regex.finditer(text)]
    firsts: list[int] = []
    for n in seq:
        if n not in firsts:
            firsts.append(n)
    ok = firsts == list(range(1, max(firsts) + 1)) if firsts else False
    return {"callouts": seq, "first_appearance": firsts, "strict_sequential": ok}


def check_figures_markdown(text: str) -> dict:
    lines = text.splitlines()
    result = {}
    for n in range(1, 7):
        mention = None
        for i, line in enumerate(lines):
            if FIG_IMAGE.match(line.strip()):
                continue
            if re.search(rf"\bFigure\s+{n}(?!\d)", line) and not FIG_CAPTION.match(line.strip()):
                mention = i
                break
        img = None
        for i in range(mention + 1 if mention is not None else 0, len(lines)):
            if FIG_IMAGE.match(lines[i].strip()):
                img = i
                break
        cap = None
        for i in range(img if img is not None else 0, len(lines)):
            m = FIG_CAPTION.match(lines[i].strip())
            if m and int(m.group(1)) == n:
                cap = i
                break
        result[n] = {
            "mention_line": mention, "image_line": img, "caption_line": cap,
            "image_after_mention": (mention is not None and img is not None and img > mention),
            "caption_after_image": (img is not None and cap is not None and cap > img),
        }
    return result


def check_figures_docx() -> dict:
    if not DOCX.exists():
        return {"available": False}
    try:
        from lxml import etree
    except Exception:  # pragma: no cover
        return {"available": False, "reason": "lxml unavailable"}
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    import zipfile
    with zipfile.ZipFile(DOCX) as z:
        root = etree.fromstring(z.read("word/document.xml"))
    paras = root.findall(".//w:body/w:p", ns)
    kinds = []
    for p in paras:
        txt = "".join(t.text or "" for t in p.findall(".//w:t", ns))
        has_draw = p.find(".//w:drawing", ns) is not None
        m = re.match(r"^\s*Figure\s+(\d+)\.", txt)
        kinds.append((int(m.group(1)) if m else None, has_draw, txt[:40]))
    result = {}
    for n in range(1, 7):
        cap_idx = next((i for i, k in enumerate(kinds) if k[0] == n), None)
        draw_idx = next((i for i, k in enumerate(kinds) if k[1]), None)
        # find the drawing that belongs to this figure: nearest drawing before/after caption
        cand = [i for i, k in enumerate(kinds) if k[1] and cap_idx is not None
                and abs(i - cap_idx) <= 2]
        result[n] = {
            "caption_index": cap_idx,
            "drawing_near_caption": bool(cand),
            "drawing_offset": (cand[0] - cap_idx) if cand else None,
        }
    return {"available": True, "figures": result,
            "all_inline": all(v["drawing_near_caption"] for v in result.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, do not write")
    args = ap.parse_args()

    man = MAN.read_text()
    sup = SUP.read_text()
    eq_main = check_equations(man, EQ_MAIN_RE)
    eq_sup = check_equations(sup, EQ_SUP_RE)
    figs = check_figures_markdown(man)

    if args.check:
        report = {"mode": "check", "equations_main": eq_main, "equations_supp": eq_sup,
                  "figures_markdown": figs, "figures_docx": check_figures_docx()}
        print(json.dumps(report, indent=2))
        ok = (eq_main["strict_sequential"] and eq_sup["strict_sequential"]
              and all(v["image_after_mention"] and v["caption_after_image"] for v in figs.values()))
        raise SystemExit(0 if ok else 1)

    refs = renumber_references()
    tables = renumber_tables()

    man = MAN.read_text()
    sup = SUP.read_text()
    report = {
        "mode": "apply",
        "references": {"first_appearance": refs["order"], "permutation": refs["perm"]},
        "supplementary_tables": {"first_appearance": tables["order"],
                                 "permutation": tables["perm"]},
        "equations_main": check_equations(man, EQ_MAIN_RE),
        "equations_supp": check_equations(sup, EQ_SUP_RE),
        "figures_markdown": figs,
        "figures_docx": check_figures_docx(),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "references_strict": report["references"]["first_appearance"] == list(range(1, 61)),
        "tables_strict": report["supplementary_tables"]["first_appearance"] == list(range(1, 19)),
        "equations_main_strict": report["equations_main"]["strict_sequential"],
        "equations_supp_strict": report["equations_supp"]["strict_sequential"],
        "figures_inline": all(v["image_after_mention"] and v["caption_after_image"]
                              for v in figs.values()),
        "report": str(REPORT.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
