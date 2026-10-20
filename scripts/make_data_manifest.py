"""Generate download/DATA_MANIFEST.md with SHA-256 provenance for imported raw data."""

import hashlib
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build(source_archive, source_sha, dest_dir, rel_root, out_path, notes=""):
    rows = []
    for dirpath, _, filenames in os.walk(dest_dir):
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, rel_root)
            rows.append((rel, os.path.getsize(full), sha256(full)))
    rows.sort()
    total = sum(r[1] for r in rows)
    lines = []
    lines.append("# DATA MANIFEST\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Purpose: immutable raw provenance for imported baseline data.\n")
    lines.append("## Source")
    lines.append(f"- Source archive: `{source_archive}`")
    lines.append(f"- Source SHA-256: `{source_sha}`")
    lines.append(f"- Imported to: `{os.path.relpath(dest_dir, PROJECT_ROOT)}`")
    lines.append(f"- Files: {len(rows)} · Total bytes: {total}\n")
    if notes:
        lines.append("## Notes")
        lines.append(notes + "\n")
    lines.append("## File hashes (SHA-256)")
    lines.append("| Relative path | Bytes | SHA-256 |")
    lines.append("|---|---:|---|")
    for rel, size, digest in rows:
        lines.append(f"| `{rel}` | {size} | `{digest}` |")
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return len(rows), total


if __name__ == "__main__":
    archive = sys.argv[1]
    source_sha = sys.argv[2]
    dest = os.path.join(PROJECT_ROOT, "download", "baseline_v0007")
    rel_root = os.path.join(dest, "nanopore-da-pinn-v0.007")
    out = os.path.join(PROJECT_ROOT, "download", "DATA_MANIFEST.md")
    notes = (
        "This directory is immutable raw provenance. It is intentionally git-ignored "
        "(`download/`). Do not edit files here; make working copies under `scripts/sg_baseline/`. "
        "Historical note from the package: the v0.003 SG flux reversed the drift sign; only "
        "v0.007 outputs are valid reference data."
    )
    n, total = build(archive, source_sha, dest, rel_root, out, notes)
    print(f"wrote {out}: {n} files, {total} bytes")
