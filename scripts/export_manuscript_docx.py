#!/usr/bin/env python3
"""Export the canonical markdown manuscripts to review-ready DOCX.

All mathematics is emitted as **native, editable Word OMML** via
``scripts/omml_math.py`` (pandoc ``$...$`` -> ``<m:oMath>``); there are no
equation images.  Display equations become numbered (1)--(8) paragraphs built
with center/right tab stops.  Figures are placed inline immediately after the
paragraph that first cites them, followed by their caption.  Tables become real
Word tables.  Prose numbers use a true minus sign and en/em dashes.
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_TAB_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from PIL import Image

import omml_math

ROOT = Path(__file__).resolve().parents[1]
JOBS = [
    (ROOT / "manuscript" / "Manuscript.md",
     ROOT / "manuscript" / "Manuscript_PINO_clean.docx", True, True),
    (ROOT / "manuscript" / "Supplementary_Materials.md",
     ROOT / "manuscript" / "Supplementary_Materials_PINO.docx", False, False),
]

DASH_EM = "\u2014"
DASH_EN = "\u2013"
MINUS = "\u2212"

# Tokens that must not be split: bold, italic, code, inline math, links.
TOKEN = re.compile(
    r"(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`|\$[^$\n]+\$|\[[^\]]+\]\([^)]+\))"
)
SUPSUB = re.compile(r"(?<=[A-Za-z0-9\)\]\}])([\^_])\{?([A-Za-z0-9+\-]+)\}?")
DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE = re.compile(r"\$([^$\n]+)\$")


def normalize_math(tex: str) -> str:
    return " ".join(tex.split())


def prose_fix(text: str) -> str:
    text = text.replace("---", DASH_EM)
    text = text.replace("--", DASH_EN)
    text = text.replace("<=", "\u2264").replace(">=", "\u2265")
    text = re.sub(r"(?<![\w\u2013\u2014])-(?=[\d.])", MINUS, text)
    return text


def add_prose(paragraph, text: str) -> None:
    text = prose_fix(text)
    cursor = 0
    for m in SUPSUB.finditer(text):
        if m.start() > cursor:
            paragraph.add_run(text[cursor:m.start()])
        run = paragraph.add_run(m.group(2))
        if m.group(1) == "^":
            run.font.superscript = True
        else:
            run.font.subscript = True
        cursor = m.end()
    if cursor < len(text):
        paragraph.add_run(text[cursor:])


def add_runs(paragraph, text: str) -> None:
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
        elif part.startswith("$") and part.endswith("$"):
            paragraph._p.append(omml_math.inline_element(normalize_math(part[1:-1])))
        elif part.startswith("*") and part.endswith("*"):
            paragraph.add_run(part[1:-1]).italic = True
        elif part.startswith("[") and "](" in part:
            label, url = part[1:-1].split("](", 1)
            paragraph.add_run(label)
            paragraph.add_run(f" ({url})")
        else:
            add_prose(paragraph, part)


def add_equation(doc: Document, tex: str, number: str) -> None:
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    pf.tab_stops.add_tab_stop(Inches(3.25), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
    run = p.add_run()
    run.add_tab()
    p._p.append(omml_math.inline_element(normalize_math(tex)))
    run = p.add_run()
    run.add_tab()
    run.add_text(f"({number})")


def add_table(doc: Document, rows: list[list[str]], size: float = 8.5) -> None:
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in r)]
    if not rows:
        return
    cols = max(len(r) for r in rows)
    table = doc.add_table(rows=0, cols=cols)
    table.style = "Table Grid"
    for ri, row in enumerate(rows):
        cells = table.add_row().cells
        for ci in range(cols):
            text = row[ci] if ci < len(row) else ""
            para = cells[ci].paragraphs[0]
            add_runs(para, text.strip())
            for run in para.runs:
                run.font.size = Pt(size)
                if ri == 0:
                    run.bold = True
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)


def add_figure(doc: Document, path: Path) -> None:
    with Image.open(path) as im:
        w_px, h_px = im.size
    width_in = min(6.4, w_px / 200.0)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Inches(width_in))


def warm_document(text: str) -> None:
    inline = [normalize_math(m.group(1)) for m in INLINE.finditer(DISPLAY.sub(" ", text))]
    display = [normalize_math(m.group(1)) for m in DISPLAY.finditer(text)]
    omml_math.warm(inline + display, [])


def build(src: Path, dst: Path, with_figures: bool, numbered_eq: bool) -> dict:
    doc = Document()
    for section in doc.sections:
        section.left_margin = section.right_margin = Inches(1.0)
        section.top_margin = section.bottom_margin = Inches(1.0)
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    text = src.read_text()
    warm_document(text)
    lines = text.splitlines()
    stats = {"paragraphs": 0, "tables": 0, "equations": 0, "figures": 0, "math": 0}
    i = 0
    eqn = 0
    in_refs = False
    refn = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped in ("---", "***", "___"):
            i += 1
            continue
        if not stripped:
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            heading = doc.add_heading(level=min(level, 4)) if level > 1 else doc.add_heading(level=0)
            add_runs(heading, m.group(2))
            if m.group(2).strip().lower() == "references":
                in_refs = True
                refn = 0
            i += 1
            continue
        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)\s*$", stripped)
        if m:
            fig = (src.parent / m.group(1)).resolve()
            if with_figures and fig.exists():
                add_figure(doc, fig)
                stats["figures"] += 1
            i += 1
            continue
        if stripped == "$$":
            j = i + 1
            buf = []
            while j < len(lines) and lines[j].strip() != "$$":
                buf.append(lines[j])
                j += 1
            eqn += 1
            number = (f"S{eqn}" if not numbered_eq else str(eqn))
            add_equation(doc, "\n".join(buf), number)
            stats["equations"] += 1
            i = j + 1
            continue
        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append([c for c in lines[i].strip().strip("|").split("|")])
                i += 1
            add_table(doc, block)
            stats["tables"] += 1
            continue
        m = re.match(r"^\*\*Figure (\d+)\.\*\*\s*(.*)$", stripped)
        if m:
            cap = doc.add_paragraph()
            cap.paragraph_format.space_after = Pt(10)
            add_runs(cap, stripped)
            for run in cap.runs:
                run.font.size = Pt(9)
            i += 1
            continue
        m = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        if m and not re.match(r"^\s*\*\*", line):
            para = doc.add_paragraph(style="List Bullet")
            add_runs(para, m.group(2))
            stats["paragraphs"] += 1
            i += 1
            continue
        m = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if m and not line.strip().startswith("**"):
            if in_refs:
                refn += 1
                para = doc.add_paragraph()
                fmt = para.paragraph_format
                fmt.left_indent = Inches(0.42)
                fmt.first_line_indent = Inches(-0.42)
                fmt.space_after = Pt(3)
                add_runs(para, f"[{refn}] " + m.group(1))
            else:
                para = doc.add_paragraph(style="List Number")
                add_runs(para, m.group(1))
            stats["paragraphs"] += 1
            i += 1
            continue
        para = doc.add_paragraph()
        add_runs(para, stripped)
        stats["paragraphs"] += 1
        i += 1

    doc.save(dst)
    stats["math"] = omml_math.cache_stats()["inline"]
    return stats


def main() -> None:
    for src, dst, with_figures, numbered_eq in JOBS:
        stats = build(src, dst, with_figures, numbered_eq)
        print(f"{dst.name}: {stats}")
    print("done")


if __name__ == "__main__":
    main()
