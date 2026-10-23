#!/usr/bin/env python3
"""Generate the submission cover letters (Markdown + DOCX).

Produces three variants sharing one body:
  * ``Cover_Letter_CPC.{md,docx}``      - Computer Physics Communications
  * ``Cover_Letter_JCP.{md,docx}``      - Journal of Computational Physics
  * ``Cover_Letter_CPC_JCP.{md,docx}``  - joint editors-in-chief letter
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_manuscript_docx import add_runs  # noqa: E402
from docx import Document  # noqa: E402
from docx.shared import Inches, Pt  # noqa: E402

OUTDIR = ROOT / "manuscript"

JOURNALS = {
    "CPC": {
        "editors": "Computer Physics Communications",
        "pitch": (
            "This work is a natural fit for *Computer Physics Communications* because its "
            "primary deliverable is a reusable, documented, and reproducible computational "
            "package. The submission includes a CPC Program Summary, an MIT-licensed public "
            "repository, trained PyTorch and ONNX models, a serverless browser simulator, and "
            "a full reproducibility package with archived result files. The study is framed explicitly as a computational-physics instrument: its contribution is the elimination of per-state retraining for design, uncertainty-quantification, and inverse workflows, not a claim of superiority over specialized solvers."
        ),
        "positioning": (
            "The submission is under the Computer Programs in Physics (CPiP) category. The software "
            "package PINO-Nanopore (archived permanently at Zenodo DOI: 10.5281/zenodo.23487404 "
            "and deployed interactively at https://pino.statwhy.com) provides a validated, "
            "exact-boundary solution operator for steady axisymmetric Poisson--Nernst--Planck "
            "transport, whose primary scientific application is high-throughput "
            "Fisher-information and conductance-identifiability mapping for nanopore experiments."
        ),
    },
    "JCP": {
        "editors": "Journal of Computational Physics",
        "pitch": (
            "This work is a natural fit for *Journal of Computational Physics* because its "
            "core contributions are a numerical-methods result and a validated identifiability "
            "ansatz that removes subtractive cancellation in coupled Nernst--Planck fields, "
            "a controlled accuracy, ablation, and out-of-distribution study of two operator "
            "backbones against a conservative Scharfetter--Gummel reference, and a Fisher "
            "analysis validated against exact float64 finite-difference Jacobians."
        ),
        "positioning": (
            "The software package PINO-Nanopore (archived permanently at Zenodo DOI: "
            "10.5281/zenodo.23487404 and deployed interactively at https://pino.statwhy.com) "
            "provides a validated, exact-boundary solution operator "
            "for steady axisymmetric Poisson--Nernst--Planck transport, together with a "
            "controlled numerical study and an identifiability analysis for nanopore experiments."
        ),
    },
    "CPC_JCP": {
        "editors": "Computer Physics Communications / Journal of Computational Physics",
        "pitch": (
            "The work combines a numerical-methods contribution with a reusable computational "
            "package: a hard-constrained operator ansatz, a controlled accuracy and ablation "
            "study, and a documented, MIT-licensed software release that includes trained "
            "models, verification reports, and an interactive simulator."
        ),
        "positioning": (
            "The software package PINO-Nanopore (archived permanently at Zenodo DOI: "
            "10.5281/zenodo.23487404 and deployed interactively at https://pino.statwhy.com) "
            "provides a validated, exact-boundary solution operator "
            "for steady axisymmetric Poisson--Nernst--Planck transport, whose primary "
            "scientific application is high-throughput Fisher-information and "
            "conductance-identifiability mapping for nanopore experiments."
        ),
    },
}

BODY = """**Date:** 23 October 2026

**To the Editors-in-Chief,** *{editors}*

**Subject:** Submission of an original research article

Dear Editors,

I submit the original research article entitled **"Charge-balanced neural operators, conductance identifiability limits, and warm-started finite-volume solvers for nanopore ion transport"** for consideration at *{editors}*. {positioning}

**Author:** Javad Alizargar, Department of Nursing, National Taipei University of Nursing and Health Sciences, No. 365, Mingde Rd, Beitou District, Taipei City 112, Taiwan (corresponding author: 8javad@ntunhs.edu.tw; ORCID: 0000-0003-2966-1315).

**Relationship to COMPHY-D-26-00857.** An earlier manuscript by the same author, *"Exact-boundary data-assimilated physics-informed neural networks for coupled Poisson--Nernst--Planck transport at charged nanopore biointerfaces"*, was submitted to Computer Physics Communications as COMPHY-D-26-00857 (revision R1); it has not been published. That study is fixed-state and needs about 500 s per new state. The present manuscript is a separate, related study. It reuses the verified SG/FV reference and the DA-PINN benchmark comparison, but its new contributions are the parameter-conditioned operator, the charge-balanced exact-boundary reparameterization with its controlled ablation, and the identifiability analysis with the blind inverse and warm-started solver benchmark. Shared methods and benchmark material are identified explicitly in the manuscript.

**Contributions.** (i) A single multi-scale radial physics-informed DeepONet maps the four parameters (applied voltage, salinity, wall charge, and radius) to the coupled potential and ionic concentrations, with exact reservoir boundary conditions and a charge-balanced hyperbolic reconstruction that removes subtractive cancellation in net charge. It reaches 0.00089/0.00109 relative L2 error for potential/current at 0.072 ms per state after ONNX export. (ii) A Fisher-information analysis, validated against exact float64 SG/FV Jacobians, shows that a conductance-only measurement leaves wall charge practically unidentifiable for symmetric electrolytes with nearly equal diffusivities. A blind 675-realization study with exact SG/FV observations confirms this limit: a dual current-plus-near-wall-selectivity protocol reaches 1.5% median charge error, but with 73% 95% coverage at 1% noise dominated by a 1.3% cross-grid discretization bias; single-salt fits do not converge, so their intervals are reported as provisional local intervals. (iii) A reference-free warm-started SG/FV solver reproduces the cold-solve current to 5 × 10⁻¹⁴ at 0.153 s (1.54× faster than a cold forward solve), and a warm-started reversal root is 1.19× faster than cold Brent. The benchmark, timing file, simulator, and MIT-licensed archive are included.

The manuscript is original, has not been published previously, and is not under consideration elsewhere. I approve this submission and declare no competing interests.

**Data and code availability.** The MIT-licensed package, including the 675 inverse records and the 120-case timing benchmark, is at https://github.com/Javad-Alizargar/nanopore-pino-surrogate. The archived release is at Zenodo DOI: 10.5281/zenodo.23487404, and the simulator is at https://pino.statwhy.com/.

Thank you for considering this manuscript. I look forward to your response and would be glad to provide any additional information.

Sincerely,

**Javad Alizargar**
Department of Nursing, National Taipei University of Nursing and Health Sciences
8javad@ntunhs.edu.tw | ORCID: 0000-0003-2966-1315
"""


def build_docx(text: str, docx_path: Path) -> int:
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.paragraph_format.space_before = Pt(0)
    style.paragraph_format.space_after = Pt(3)
    style.paragraph_format.line_spacing = 1.0
    doc.styles["Heading 1"].paragraph_format.space_before = Pt(0)
    doc.styles["Heading 1"].paragraph_format.space_after = Pt(6)
    for block in text.split("\n"):
        line = block.rstrip()
        if not line:
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
            continue
        p = doc.add_paragraph()
        add_runs(p, line)
    doc.save(docx_path)
    return len(doc.paragraphs)


def main() -> None:
    for key, cfg in JOURNALS.items():
        letter = BODY.replace("{editors}", cfg["editors"]).replace("{pitch}", cfg["pitch"]).replace("{positioning}", cfg["positioning"])
        md = OUTDIR / f"Cover_Letter_{key}.md"
        docx = OUTDIR / f"Cover_Letter_{key}.docx"
        md.write_text(letter)
        n = build_docx(letter, docx)
        text = "\n".join(p.text for p in Document(docx).paragraphs)
        leak = [tok for tok in ("$$", "\\", "mathcal", "{", "}") if tok in text]
        print(f"wrote {md.relative_to(ROOT)} ({md.stat().st_size} B)")
        print(f"wrote {docx.relative_to(ROOT)} ({docx.stat().st_size} B, {n} paragraphs, leak={leak})")


if __name__ == "__main__":
    main()
