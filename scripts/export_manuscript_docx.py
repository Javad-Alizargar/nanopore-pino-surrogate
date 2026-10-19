#!/usr/bin/env python3
"""Export the canonical markdown manuscripts to review-ready DOCX.

Display equations are rendered to images with matplotlib mathtext; inline math
is converted to Unicode. All tables become real Word tables and the six main
figures are embedded with their captions.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "manuscript" / "figures"
JOBS = [
    (ROOT / "manuscript" / "Manuscript.md", ROOT / "manuscript" / "Manuscript_PINO_clean.docx", True),
    (ROOT / "manuscript" / "Supplementary_Materials.md", ROOT / "manuscript" / "Supplementary_Materials_PINO.docx", False),
]

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ", "chi": "χ",
    "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
    "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ",
    "Psi": "Ψ", "Omega": "Ω",
}
SYMBOLS = {
    "times": "×", "cdot": "·", "pm": "±", "mp": "∓", "le": "≤", "leq": "≤",
    "ge": "≥", "geq": "≥", "to": "→", "approx": "≈", "infty": "∞",
    "partial": "∂", "nabla": "∇", "sum": "Σ", "int": "∫", "equiv": "≡",
    "propto": "∝", "sim": "∼", "odot": "⊙", "ell": "ℓ", "lVert": "‖",
    "rVert": "‖", "vert": "|", "langle": "⟨", "rangle": "⟩", "circ": "°",
    "sqrt": "√", "quad": " ", "qquad": "  ", "mathrm": "", "text": "",
    "mathcal": "", "mathbf": "", "bf": "", "left": "", "right": "",
    "big": "", "Big": "", "bigg": "", "Bigg": "", "overline": "", "bar": "",
    "hat": "", "tilde": "", "frac": "", "tfrac": "", "dfrac": "", "!": "",
    ",": " ", ";": " ", ":": " ",
}


def inline_math(tex: str) -> str:
    out = tex
    out = re.sub(r"\\(?:lVert|rVert)", "‖", out)
    out = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", out)
    out = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", out)
    for name, ch in sorted(GREEK.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(r"\\" + name + r"(?![a-zA-Z])", ch, out)
    for name, ch in sorted(SYMBOLS.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(r"\\" + re.escape(name) + r"(?![a-zA-Z])", ch, out)
    out = re.sub(r"\\[a-zA-Z]+", "", out)
    out = out.replace("\\", "")
    out = out.replace("{", "").replace("}", "")
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def sanitize_tex(tex: str) -> str:
    out = tex.strip()
    out = re.sub(r"\\lVert|\\rVert", r"\|", out)
    out = out.replace(r"\!", "")
    out = re.sub(r"\\big[glr]?", "", out)
    out = re.sub(r"\\Big[glr]?", "", out)
    out = out.replace(r"\tfrac", r"\frac")
    return out


_eq_counter = [0]


def render_equation(tex: str, tmp: Path) -> Path | None:
    _eq_counter[0] += 1
    path = tmp / f"eq_{_eq_counter[0]:03d}.png"
    clean = sanitize_tex(tex)
    try:
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, f"${clean}$", fontsize=12, color="black")
        fig.savefig(path, dpi=220, bbox_inches="tight", transparent=False,
                    facecolor="white", pad_inches=0.05)
        plt.close(fig)
        return path
    except Exception:
        return None


def add_equation(doc: Document, tex: str, tmp: Path) -> None:
    path = render_equation(tex, tmp)
    if path is None:
        p = doc.add_paragraph(inline_math(tex))
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        return
    with Image.open(path) as im:
        w_px, h_px = im.size
    width_in = w_px / 220.0
    if width_in > 6.1:
        width_in = 6.1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width_in))


TOKEN = re.compile(r"(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`|\$[^$\n]+\$|\[[^\]]+\]\([^)]+\))")


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
            paragraph.add_run(inline_math(part[1:-1]))
        elif part.startswith("*") and part.endswith("*"):
            paragraph.add_run(part[1:-1]).italic = True
        elif part.startswith("[") and "](" in part:
            label, url = part[1:-1].split("](", 1)
            run = paragraph.add_run(label)
            run.font.color.rgb = None
            paragraph.add_run(f" ({url})")
        else:
            paragraph.add_run(part)


def add_table(doc: Document, rows: list[list[str]]) -> None:
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
                run.font.size = Pt(8.5)
                if ri == 0:
                    run.bold = True
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)


def find_fig(n: int) -> Path | None:
    matches = sorted(FIGDIR.glob(f"fig{n}_*.png"))
    return matches[0] if matches else None


def build(src: Path, dst: Path, with_figures: bool) -> dict:
    doc = Document()
    for section in doc.sections:
        section.left_margin = section.right_margin = Inches(1.0)
        section.top_margin = section.bottom_margin = Inches(1.0)
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    lines = src.read_text().splitlines()
    tmp = Path(tempfile.mkdtemp(prefix="docx_eq_"))
    i = 0
    stats = {"paragraphs": 0, "tables": 0, "equations": 0, "figures": 0}
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
            i += 1
            continue
        if stripped == "$$":
            j = i + 1
            buf = []
            while j < len(lines) and lines[j].strip() != "$$":
                buf.append(lines[j])
                j += 1
            add_equation(doc, "\n".join(buf), tmp)
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
        if m and with_figures:
            n = int(m.group(1))
            fig = find_fig(n)
            if fig:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(str(fig), width=Inches(6.1))
                stats["figures"] += 1
            cap = doc.add_paragraph()
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
    return stats


def main() -> None:
    for src, dst, with_figures in JOBS:
        stats = build(src, dst, with_figures)
        print(f"{dst.name}: {stats}")
    print("done")


if __name__ == "__main__":
    main()
