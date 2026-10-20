#!/usr/bin/env python3
"""Convert LaTeX math to native Word OMML using pandoc, with an on-disk cache.

Word's native, editable equation format is OMML (``<m:oMath>`` inline and
``<m:oMathPara>`` for display equations).  Pandoc renders ``$...$`` and
``$$...$$`` to exactly those elements, so we batch every unique expression from
the manuscript through a single pandoc ``markdown -> docx`` call, parse the
generated ``word/document.xml``, and map each expression to its OMML element.

Results are cached in ``manuscript/cache/omml_cache.json`` so repeat DOCX builds
do not re-invoke pandoc.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import lxml.etree as ET
from docx.oxml import parse_xml
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "manuscript" / "cache" / "omml_cache.json"
PANDOC = shutil.which("pandoc") or "/opt/homebrew/bin/pandoc"

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

_CACHE: dict[str, dict[str, str]] | None = None


def _load_cache() -> dict[str, dict[str, str]]:
    global _CACHE
    if _CACHE is None:
        if CACHE_PATH.exists():
            _CACHE = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        else:
            _CACHE = {"inline": {}, "display": {}}
        _CACHE.setdefault("inline", {})
        _CACHE.setdefault("display", {})
    return _CACHE


def _save_cache() -> None:
    cache = _load_cache()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )


def _pandoc_available() -> bool:
    return bool(PANDOC) and Path(PANDOC).exists()


def _run_pandoc(markdown: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "batch.md"
        src.write_text(markdown, encoding="utf-8")
        out = tmp_path / "batch.docx"
        subprocess.run(
            [PANDOC, "-f", "markdown", "-t", "docx", "-o", str(out), str(src)],
            check=True,
            capture_output=True,
        )
        with zipfile.ZipFile(out) as zf:
            return zf.read("word/document.xml")


def _paragraph_text(p) -> str:
    return "".join(t.text or "" for t in p.iter(W + "t"))


def _first(p, tag: str):
    for el in p.iter(tag):
        return el
    return None


def _convert_batch(kinds: dict[str, list[str]]) -> None:
    """kinds maps 'inline'/'display' -> list of missing LaTeX strings."""
    cache = _load_cache()
    lines: list[str] = []
    inline_ids: list[str] = list(dict.fromkeys(kinds.get("inline", [])))
    display_ids: list[str] = list(dict.fromkeys(kinds.get("display", [])))
    for i, tex in enumerate(inline_ids):
        lines.append(f"MIN{i} ${tex}$")
        lines.append("")
    for i, tex in enumerate(display_ids):
        lines.append(f"DMD{i}")
        lines.append("")
        lines.append(f"$${tex}$$")
        lines.append("")

    xml = _run_pandoc("\n".join(lines))
    root = ET.fromstring(xml)

    inline_map = {f"MIN{i}": tex for i, tex in enumerate(inline_ids)}
    display_map = {f"DMD{i}": tex for i, tex in enumerate(display_ids)}

    pending_display: str | None = None
    found_inline: set[str] = set()
    found_display: set[str] = set()

    for p in root.iter(W + "p"):
        text = _paragraph_text(p).strip()
        marker = text.split()[0] if text else ""
        if marker in inline_map:
            omath = _first(p, M + "oMath")
            if omath is not None:
                cache["inline"][inline_map[marker]] = ET.tostring(
                    omath, encoding="unicode"
                )
                found_inline.add(marker)
            continue
        if marker in display_map:
            pending_display = marker
            continue
        if pending_display is not None:
            opara = _first(p, M + "oMathPara")
            if opara is None:
                opara = _first(p, M + "oMath")
            if opara is not None:
                cache["display"][display_map[pending_display]] = ET.tostring(
                    opara, encoding="unicode"
                )
                found_display.add(pending_display)
                pending_display = None

    missing = [inline_map[k] for k in inline_map if k not in found_inline]
    missing += [display_map[k] for k in display_map if k not in found_display]
    if missing:
        raise RuntimeError(f"pandoc did not return OMML for: {missing}")
    _save_cache()


def warm(inline: list[str], display: list[str]) -> None:
    cache = _load_cache()
    miss_inline = [t for t in dict.fromkeys(inline) if t not in cache["inline"]]
    miss_display = [t for t in dict.fromkeys(display) if t not in cache["display"]]
    if not miss_inline and not miss_display:
        return
    if not _pandoc_available():
        raise RuntimeError("pandoc is required to render Word equations")
    _convert_batch({"inline": miss_inline, "display": miss_display})


def _element(kind: str, latex: str):
    cache = _load_cache()
    if latex not in cache[kind]:
        warm([latex] if kind == "inline" else [],
             [latex] if kind == "display" else [])
    return parse_xml(cache[kind][latex].encode("utf-8"))


def inline_element(latex: str):
    """Return an ``<m:oMath>`` element for inline placement in a paragraph."""
    return _element("inline", latex)


def display_element(latex: str):
    """Return an ``<m:oMathPara>`` element for a display equation paragraph."""
    return _element("display", latex)


def cache_stats() -> dict[str, int]:
    cache = _load_cache()
    return {"inline": len(cache["inline"]), "display": len(cache["display"])}
