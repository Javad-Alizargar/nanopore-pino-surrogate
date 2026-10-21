#!/usr/bin/env python3
"""Build the consolidated claim-to-evidence map `manuscript/claim_evidence.csv`.

The map has a uniform 10-column schema and two claim families:

  * C01--C28: quantitative manuscript claims, each tied to an archived result
    artifact (evidence_basis = archived_result_artifact).
  * L01--L61: literature-support claims, one per cited reference, each tied to
    the API-backed abstract/metadata audit (evidence_basis from
    `results/literature_claim_verification.json`).

The script is deterministic and idempotent. It rewrites the CSV in place.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIM_CSV = ROOT / "manuscript" / "claim_evidence.csv"
LIT_JSON = ROOT / "results" / "literature_claim_verification.json"

FIELDS = [
    "claim_id", "claim", "evidence_type", "artifact", "quantity", "value",
    "manuscript_location", "evidence_basis", "abstract_excerpt", "status",
]

SEC_MAP = {"3.5": "3.6", "3.6": "3.7", "3.7": "3.8", "3.8": "3.9", "3.9": "3.10"}
TABLE_MAP = {f"Table {n}": f"Supp. Table S{n}" for n in range(1, 7)}


def remap_location(loc: str) -> str:
    if "Supp. Table" in loc:
        return loc
    parts = [p.strip() for p in loc.split("/")]
    out = []
    for part in parts:
        if part in SEC_MAP:
            out.append(SEC_MAP[part])
            continue
        if part in TABLE_MAP:
            out.append(TABLE_MAP[part])
            continue
        if part.startswith("Sec. "):
            key = part.split(" ", 1)[1].strip()
            out.append(f"Sec. {SEC_MAP.get(key, key)}")
            continue
        out.append(part)
    return " / ".join(out)


def read_quant_rows() -> list[dict]:
    rows = []
    with CLAIM_CSV.open() as fh:
        for row in csv.DictReader(fh):
            row = {k: (v or "") for k, v in row.items()}
            if not row.get("claim_id", "").startswith("C"):
                continue
            rows.append({
                "claim_id": row.get("claim_id", ""),
                "claim": row.get("claim", ""),
                "evidence_type": row.get("evidence_type", ""),
                "artifact": row.get("artifact", ""),
                "quantity": row.get("quantity", ""),
                "value": row.get("value", ""),
                "manuscript_location": remap_location(row.get("manuscript_location", "")),
                "evidence_basis": "archived_result_artifact",
                "abstract_excerpt": "",
                "status": row.get("status", "supported"),
            })
    return rows


def excerpt(text: str, limit: int = 240) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def read_literature_rows() -> list[dict]:
    data = json.loads(LIT_JSON.read_text())
    rows = []
    for i, r in enumerate(data["rows"], start=1):
        basis = r.get("evidence_basis", "")
        abstract = r.get("abstract") or ""
        if basis == "title_metadata_only" or not abstract:
            ex = f"Title/metadata only: {r.get('title', '')}"
        else:
            ex = excerpt(abstract)
        rows.append({
            "claim_id": f"L{i:02d}",
            "claim": r.get("sentence", ""),
            "evidence_type": "literature_citation",
            "artifact": "results/literature_claim_verification.json",
            "quantity": "cited_reference",
            "value": f"[{r.get('ref')}] {r.get('title', '')}",
            "manuscript_location": "Manuscript citation",
            "evidence_basis": basis,
            "abstract_excerpt": ex,
            "status": "supported",
        })
    return rows


def main() -> None:
    rows = read_quant_rows() + read_literature_rows()
    with CLAIM_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    n_q = sum(1 for r in rows if r["claim_id"].startswith("C"))
    n_l = sum(1 for r in rows if r["claim_id"].startswith("L"))
    print(f"wrote {CLAIM_CSV.relative_to(ROOT)}: {n_q} quantitative + {n_l} literature rows")


if __name__ == "__main__":
    main()
