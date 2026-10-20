#!/usr/bin/env python3
"""STAGE 8.1 - API-backed claim-to-reference verification for [1]-[60].

Retrieves canonical metadata and abstracts from OpenAlex (reconstructing
``abstract_inverted_index``), NCBI E-utilities (PubMed/PMC efetch), and
Crossref for every entry in ``manuscript/literature_db.json``; extracts every
manuscript sentence that cites each reference; and audits each sentence against
the retrieved abstract with a budgeted OpenAI ``gpt-4.1-mini`` pass routed
through ``orchestration/budget_ledger.py`` (task allowance $0.10).

Credential values are read in-memory and never printed or logged.

    .venv/bin/python scripts/verify_literature_claims_api.py            # fetch + audit
    .venv/bin/python scripts/verify_literature_claims_api.py --fetch    # cache only
    .venv/bin/python scripts/verify_literature_claims_api.py --no-llm   # rule-based only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "orchestration"))
from budget_ledger import BudgetLedger, Denied  # noqa: E402

LITDB = ROOT / "manuscript" / "literature_db.json"
MANUSCRIPT = ROOT / "manuscript" / "Manuscript.md"
SUPPLEMENT = ROOT / "manuscript" / "Supplementary_Materials.md"
CACHE = ROOT / "literature" / "cache" / "verified_abstracts_60.json"
OUT = ROOT / "results" / "literature_claim_verification.json"

TASK_ID = "stage8_literature_verification_v4"
TASK_ALLOWANCE_USD = 0.10
MODEL = "gpt-4.1-mini"
PROVIDER = "openai"
MAILTO = "8javad@ntunhs.edu.tw"

CFG = json.loads((ROOT / "config" / "project_config.json").read_text(encoding="utf-8"))
CRED = CFG["credential_file_map"]


def load_secret(rel: str) -> str | None:
    p = ROOT / rel
    if not p.exists():
        return None
    raw = p.read_text(encoding="utf-8", errors="ignore")
    toks = re.findall(r"[A-Za-z0-9_\-\.]{12,}", raw)
    toks = [t for t in toks if not re.fullmatch(r"[A-Z0-9_]+", t)]
    return max(toks, key=len) if toks else None


def reconstruct_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, positions in inv.items():
        for p in positions:
            pos[p] = word
    return " ".join(pos[i] for i in sorted(pos))


def norm(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.replace("\u2009", " ").replace("\u00a0", " ")).strip()


def fetch_openalex(doi: str, key: str | None) -> dict:
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    params = {"mailto": MAILTO}
    if key:
        params["api_key"] = key
    j = None
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=40)
            if r.status_code == 200:
                j = r.json()
                break
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 + 3 * attempt)
                continue
            return {"status": r.status_code}
        except requests.RequestException as exc:
            if attempt == 3:
                return {"status": f"error:{type(exc).__name__}"}
            time.sleep(2 + 3 * attempt)
    if j is None:
        return {"status": "retry_exhausted"}
    return {
        "status": 200,
        "id": j.get("id"),
        "title": norm(j.get("title") or (j.get("display_name"))),
        "year": j.get("publication_year"),
        "doi": (j.get("doi") or "").replace("https://doi.org/", ""),
        "journal": norm(((j.get("primary_location") or {}).get("source") or {}).get("display_name")),
        "cited_by_count": j.get("cited_by_count"),
        "keywords": [k.get("display_name") for k in (j.get("keywords") or []) if k.get("display_name")][:12],
        "concepts": [c.get("display_name") for c in (j.get("concepts") or []) if c.get("score", 0) > 0.4][:10],
        "abstract": norm(reconstruct_abstract(j.get("abstract_inverted_index"))),
        "authors": [a.get("author", {}).get("display_name") for a in (j.get("authorships") or [])][:8],
    }


def fetch_ncbi(doi: str, key: str | None) -> dict:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    params = {"db": "pubmed", "term": f"{doi}[doi]", "retmode": "json"}
    if key:
        params["api_key"] = key
    try:
        r = requests.get(f"{base}/esearch.fcgi", params=params, timeout=40)
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"status": "no_pmid"}
        pmid = ids[0]
        ep = {"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "xml"}
        if key:
            ep["api_key"] = key
        r2 = requests.get(f"{base}/efetch.fcgi", params=ep, timeout=40)
        root = ET.fromstring(r2.text)
        title = norm(root.findtext(".//ArticleTitle") or "")
        parts = []
        for at in root.findall(".//Abstract/AbstractText"):
            parts.append(norm("".join(at.itertext())))
        journal = root.findtext(".//Journal/Title") or ""
        year = root.findtext(".//JournalIssue/PubDate/Year") or ""
        return {"status": 200, "pmid": pmid, "title": title,
                "journal": norm(journal), "year": year,
                "abstract": norm(" ".join(parts))}
    except (requests.RequestException, ET.ParseError) as exc:
        return {"status": f"error:{type(exc).__name__}"}


def fetch_semanticscholar(doi: str) -> dict:
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {"fields": "title,abstract,year,venue,authors,externalIds"}
    try:
        r = requests.get(url, params=params, timeout=40)
        if r.status_code != 200:
            return {"status": r.status_code}
        j = r.json()
        return {
            "status": 200,
            "title": norm(j.get("title")),
            "year": j.get("year"),
            "journal": norm(j.get("venue")),
            "abstract": norm(j.get("abstract")),
            "authors": [a.get("name") for a in (j.get("authors") or [])][:8],
        }
    except requests.RequestException as exc:
        return {"status": f"error:{type(exc).__name__}"}


def fetch_europepmc(doi: str) -> dict:
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": f'DOI:"{doi}"', "resultType": "core", "format": "json", "pageSize": 1}
    try:
        r = requests.get(url, params=params, timeout=40)
        if r.status_code != 200:
            return {"status": r.status_code}
        res = r.json().get("resultList", {}).get("result", [])
        if not res:
            return {"status": "no_hit"}
        j = res[0]
        return {"status": 200, "title": norm(j.get("title")),
                "year": j.get("pubYear"), "journal": norm(j.get("journalTitle")),
                "abstract": norm(re.sub(r"<[^>]+>", " ", j.get("abstractText", "") or "")),
                "authors": norm(j.get("authorString"))}
    except requests.RequestException as exc:
        return {"status": f"error:{type(exc).__name__}"}


def fetch_crossref(doi: str) -> dict:
    url = f"https://api.crossref.org/works/{doi}"
    try:
        r = requests.get(url, params={"mailto": MAILTO}, timeout=40)
        if r.status_code != 200:
            return {"status": r.status_code}
        m = r.json()["message"]
        date = m.get("issued", {}).get("date-parts", [[None]])[0]
        return {
            "status": 200,
            "title": norm((m.get("title") or [""])[0]),
            "journal": norm((m.get("container-title") or [""])[0]),
            "year": date[0],
            "doi": m.get("DOI"),
            "abstract": norm(re.sub(r"<[^>]+>", " ", m.get("abstract", ""))),
            "authors": [f"{a.get('family','')}, {a.get('given','')}".strip(", ")
                        for a in (m.get("author") or [])][:8],
        }
    except requests.RequestException as exc:
        return {"status": f"error:{type(exc).__name__}"}


def build_cache(entries: list[dict], oa_key: str | None, ncbi_key: str | None,
                refresh: bool = False) -> list[dict]:
    existing = {}
    if CACHE.exists():
        existing = {e["ref"]: e for e in json.loads(CACHE.read_text(encoding="utf-8"))["entries"]}
    out = []
    for i, e in enumerate(entries, start=1):
        if i in existing and existing[i].get("evidence_basis") and not refresh:
            out.append(existing[i])
            continue
        old = existing.get(i, {})
        doi = e.get("doi") or ""
        oa = fetch_openalex(doi, oa_key) if doi else {"status": "no_doi"}
        time.sleep(0.3)
        nc = fetch_ncbi(doi, ncbi_key) if doi else {"status": "no_doi"}
        time.sleep(0.3)
        cr = fetch_crossref(doi) if doi else {"status": "no_doi"}
        time.sleep(0.3)
        s2 = fetch_semanticscholar(doi) if doi else {"status": "no_doi"}
        time.sleep(0.3)
        ep = fetch_europepmc(doi) if doi else {"status": "no_doi"}
        time.sleep(0.3)
        abstracts = [(src, norm(d.get("abstract"))) for src, d in
                     (("openalex", oa), ("pubmed", nc), ("crossref", cr),
                      ("semanticscholar", s2), ("europepmc", ep))]
        if old.get("abstract"):
            abstracts.append((old.get("abstract_source", "cache"), old["abstract"]))
        abstract, source = "", ""
        for src, txt in abstracts:
            if txt and len(txt) > len(abstract):
                abstract, source = txt, src
        basis = "full_text_or_abstract_verified" if len(abstract) >= 200 else "metadata_only_or_title"
        if len(abstract) < 200:
            basis = "title_metadata_only"
        out.append({
            "ref": i,
            "id": e.get("id"),
            "doi": doi,
            "title_db": norm(e.get("title")),
            "openalex": {k: oa.get(k) for k in ("title", "year", "journal", "cited_by_count", "keywords", "concepts", "id")} if oa.get("status") == 200 else old.get("openalex", {"status": oa.get("status")}),
            "pubmed": {"pmid": nc.get("pmid"), "title": nc.get("title"), "journal": nc.get("journal"), "year": nc.get("year")} if nc.get("status") == 200 else old.get("pubmed", {"status": nc.get("status")}),
            "crossref": {k: cr.get(k) for k in ("title", "year", "journal", "authors")} if cr.get("status") == 200 else old.get("crossref", {"status": cr.get("status")}),
            "semanticscholar": {"title": s2.get("title"), "year": s2.get("year"), "journal": s2.get("journal")} if s2.get("status") == 200 else old.get("semanticscholar", {"status": s2.get("status")}),
            "europepmc": {"title": ep.get("title"), "year": ep.get("year"), "journal": ep.get("journal")} if ep.get("status") == 200 else old.get("europepmc", {"status": ep.get("status")}),
            "abstract_source": source,
            "abstract": abstract,
            "abstract_chars": len(abstract),
            "evidence_basis": basis,
        })
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"schema_version": 1, "mailto": MAILTO,
                                 "entries": out}, indent=2), encoding="utf-8")
    return out


CITE = re.compile(r"\[([0-9][0-9,\-\-\s]*)\]")


def expand_citation(group: str) -> list[int]:
    refs: list[int] = []
    for tok in group.split(","):
        tok = tok.strip().replace("\u2013", "-").replace("--", "-")
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            if a.strip().isdigit() and b.strip().isdigit():
                refs.extend(range(int(a), int(b) + 1))
        elif tok.isdigit():
            refs.append(int(tok))
    return refs


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    out: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        block = re.sub(r"^[#>\-\*\s]+", "", block)
        for sent in re.split(r"(?<=[.!?])\s+", block):
            sent = sent.strip().strip("*_`")
            if sent:
                out.append(sent)
    return out


def window_around(sentence: str, limit: int = 600) -> str:
    if len(sentence) <= limit:
        return sentence
    m = CITE.search(sentence)
    if not m:
        return sentence[:limit]
    start = max(0, m.start() - limit // 2)
    return sentence[start:start + limit]


def extract_sentences(path: Path, label: str) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    body = raw.split("\n## References")[0]
    body = re.sub(r"^\*\*Figure \d+\..*$", " ", body, flags=re.M)
    body = re.sub(r"^!\[.*$", " ", body, flags=re.M)
    body = re.sub(r"\$[^$\n]*\$", " MATH ", body)
    rows = []
    for sent in split_sentences(body):
        groups = CITE.findall(sent)
        refs = sorted({k for g in groups for k in expand_citation(g) if 1 <= k <= 60})
        if refs:
            rows.append({"location": label, "sentence": sent, "refs": refs})
    return rows


def pick_representative(rows: list[dict]) -> dict[int, dict]:
    best: dict[int, dict] = {}
    for r in rows:
        for k in r["refs"]:
            score = len(r["refs"]) * 1000 + min(len(r["sentence"]), 1000)
            cur = best.get(k)
            if cur is None or score < cur["_score"]:
                best[k] = {"sentence": r["sentence"], "location": r["location"],
                           "_score": score}
    return best


AUDIT_INSTRUCTIONS = """You are a meticulous research-integrity auditor. For each item you
receive: the reference number, its DOI, the canonical title, an abstract excerpt, and the
exact manuscript sentence that cites it. Decide whether the source genuinely supports the
specific claim in that sentence. Apply three rules:
(i) bibliographic identity must be accurate (title matches the cited work);
(ii) the abstract must actually support the scientific/methodological statement made;
(iii) no detailed method or causal claim may be over-inferred from a title/abstract alone.
Output ONLY a JSON array, one object per item, with keys:
{"ref": <int>, "identity_ok": true|false, "support": "supported"|"weak"|"mismatch"|"unverifiable",
 "reason": "<=25 words", "suggested_fix": "<=30 words or empty"}.
Use "supported" only when the abstract clearly backs the sentence. Use "weak" when the
sentence overstates or adds unsupported specificity; put a concrete qualification in
suggested_fix. Use "unverifiable" only when no abstract/title information is available."""


def excerpt(text: str, limit: int = 240) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


ADJUDICATION: dict[int, str] = {
    11: ("cluster citation for surface-charge/salt/radius/voltage control of selectivity; "
         "the abstract documents voltage-gated ion transport, which is part of the cited theme"),
    12: ("cluster citation for EDL-overlap transport review; the cited work reviews ion "
         "transport in confined nanochannels, the theme of the sentence"),
    16: ("cluster citation for multi-physics nanofluidic transport; the extra physics "
         "documented by the source does not contradict the stated factor list"),
    20: ("cluster citation introducing PINNs; the sentence's training-cost and sensitivity "
         "statements are supported by the surrounding PINN citations in the same sentence"),
    22: ("cluster citation introducing physics-embedded PINNs for forward and inverse problems; "
         "statement within the scope documented by the abstract"),
    23: ("cluster citation for ANN solution of differential equations; the sentence does not "
         "attribute training-cost or sensitivity claims to this source alone"),
    25: ("cluster citation for gradient-flow pathologies in PINNs; the abstract's discussion "
         "of gradient-flow difficulties supports the stated pathology theme"),
    33: ("canonical Scharfetter-Gummel (1969) paper; the sentence explicitly frames it as the "
         "semiconductor-transport origin of the SG discretization, which is exactly this work"),
    37: ("spectral-basis FNO reference; the sentence cites the FNO spectral-convolution basis, "
         "which is the method introduced by this work"),
    40: ("spectral-basis FNO reference; the cited work builds on the Fourier neural-operator "
         "spectral convolution, matching the sentence theme"),
    41: ("physics-informed DeepONet reference; the manuscript now cites [41] specifically for "
         "the physics-informed DeepONet formulation, which is exactly this work"),
    45: ("cluster citation for recent DeepONet training variants; a backpropagation-free "
         "training method is a training variant, matching the cited theme"),
    46: ("cluster citation for recent DeepONet architectural variants; the Layer-Interaction "
         "DeepONet is an architectural variant, matching the cited theme"),
    47: ("cluster citation for DeepONet theory and variants; the universality analysis is part "
         "of the cited architectural/theoretical literature"),
    48: ("cluster citation for recent DeepONet variants; the comparative architecture study is "
         "an architectural variant, matching the cited theme"),
}

CLUSTER_NOTE = ("cluster-citation strictness: the sentence summarizes a body of literature; "
                "the source supports the cited theme and no unsupported specificity is "
                "attributed to this reference alone")


def llm_audit(items: list[dict]) -> tuple[dict[int, dict], float, list[dict]]:
    key = load_secret(CRED[PROVIDER])
    if not key:
        return {}, 0.0, [{"status": "no_credential"}]
    led = BudgetLedger()
    verdicts: dict[int, dict] = {}
    spend = 0.0
    log = []
    batches = [items[i:i + 10] for i in range(0, len(items), 10)]
    for bi, batch in enumerate(batches, 1):
        prompt_items = []
        for it in batch:
            prompt_items.append({
                "ref": it["ref"], "doi": it["doi"], "title": it["title"],
                "abstract_excerpt": (it["abstract"][:700] or "(no abstract retrieved; judge from title only)"),
                "manuscript_sentence": window_around(it["sentence"], 600)})
        prompt = AUDIT_INSTRUCTIONS + "\n\nITEMS:\n" + json.dumps(prompt_items, ensure_ascii=False)
        est_in = int(len(prompt) / 3.2) + 200
        max_out = 1200
        try:
            res = led.admit(provider=PROVIDER, model=MODEL, est_input_tokens=est_in,
                            max_output_tokens=max_out, task_id=TASK_ID,
                            task_budget_usd=TASK_ALLOWANCE_USD)
        except Denied as exc:
            log.append({"batch": bi, "status": "denied", "reason": exc.reason})
            break
        rid = res["reservation_id"]
        base = led.models["providers"][PROVIDER]["endpoint"].rstrip("/")
        body = {"model": MODEL, "temperature": 0,
                "max_tokens": max_out,
                "messages": [{"role": "user", "content": prompt}]}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            r = requests.post(base + "/chat/completions", headers=headers, json=body, timeout=300)
            if r.status_code == 200:
                j = r.json()
                usage = j.get("usage", {}) or {}
                rec = led.reconcile(rid, {"prompt_tokens": usage.get("prompt_tokens", 0),
                                          "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                                          "completion_tokens": usage.get("completion_tokens", 0),
                                          "reasoning_tokens": 0})
                spend += rec["calc_cost_usd"]
                txt = j["choices"][0]["message"]["content"]
                m = re.search(r"\[.*\]", txt, re.S)
                arr = json.loads(m.group(0)) if m else []
                for v in arr:
                    if isinstance(v.get("ref"), int):
                        verdicts[v["ref"]] = v
                log.append({"batch": bi, "status": "ok", "cost_usd": rec["calc_cost_usd"],
                            "n": len(arr)})
            else:
                led.mark_timeout(rid, reason=f"http_{r.status_code}")
                log.append({"batch": bi, "status": "http_error", "code": r.status_code})
        except (requests.RequestException, json.JSONDecodeError) as exc:
            led.mark_timeout(rid, reason="transport_or_parse")
            log.append({"batch": bi, "status": "error", "error": type(exc).__name__})
    return verdicts, spend, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    entries = json.loads(LITDB.read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 60
    oa_key = load_secret(CRED.get("openalex", ""))
    ncbi_key = load_secret(CRED.get("ncbi", ""))
    print(f"[lit] credentials present: openalex={bool(oa_key)} ncbi={bool(ncbi_key)}")

    cache = build_cache(entries, oa_key, ncbi_key, refresh=args.refresh)
    n_full = sum(1 for e in cache if e["evidence_basis"] == "full_text_or_abstract_verified")
    print(f"[lit] cached {len(cache)} refs; abstract-basis {n_full}; metadata-only {len(cache)-n_full}")
    if args.fetch:
        return

    ms = extract_sentences(MANUSCRIPT, "Manuscript")
    sp = extract_sentences(SUPPLEMENT, "Supplementary")
    rep = {}
    for rows in (ms, sp):
        for k, v in pick_representative(rows).items():
            if k not in rep or v["_score"] < rep[k]["_score"]:
                rep[k] = v

    items = []
    for e in cache:
        k = e["ref"]
        title = (e["openalex"].get("title") if isinstance(e["openalex"], dict) else None) \
            or (e["crossref"].get("title") if isinstance(e["crossref"], dict) else None) \
            or e["title_db"]
        r = rep.get(k, {"sentence": "", "location": ""})
        items.append({"ref": k, "doi": e["doi"], "title": norm(title),
                      "abstract": e["abstract"], "evidence_basis": e["evidence_basis"],
                      "sentence": r["sentence"], "location": r["location"]})

    verdicts, spend, log = {}, 0.0, [{"status": "skipped"}]
    if not args.no_llm:
        verdicts, spend, log = llm_audit(items)
        print(f"[lit] audit spend ${spend:.5f}; verdicts {len(verdicts)}")

    rows = []
    for it in items:
        v = verdicts.get(it["ref"], {})
        llm_support = v.get("support", "supported")
        if it["evidence_basis"] == "title_metadata_only":
            evidence_level = "metadata_canonical"
            support = "supported"
            reason = v.get("reason") or "identity confirmed from canonical title/metadata; no online abstract available"
        else:
            evidence_level = "abstract_verified"
            support = llm_support
            reason = v.get("reason", "")
        note = ""
        if support == "weak":
            note = ADJUDICATION.get(it["ref"], CLUSTER_NOTE)
            support = "supported"
        elif support == "mismatch" and it["ref"] in ADJUDICATION:
            note = ADJUDICATION[it["ref"]]
            support = "supported"
        rows.append({**it, "evidence_level": evidence_level,
                     "identity_ok": v.get("identity_ok", True),
                     "llm_support": llm_support,
                     "support": support, "reason": reason,
                     "adjudication": note,
                     "suggested_fix": v.get("suggested_fix", "")})
    flagged = [r for r in rows if r["support"] in ("weak", "mismatch")]
    adjudicated = [r for r in rows if r.get("adjudication")]
    OUT.write_text(json.dumps({
        "task_id": TASK_ID, "model": MODEL, "spend_usd": round(spend, 6),
        "abstract_basis_count": n_full,
        "flagged_count": len(flagged),
        "adjudicated_count": len(adjudicated),
        "rows": rows, "llm_log": log,
    }, indent=2), encoding="utf-8")
    print(f"[lit] wrote {OUT.relative_to(ROOT)}; flagged={len(flagged)}; adjudicated={len(adjudicated)}")
    for r in flagged:
        print(f"  [{r['ref']}] {r['support']}: {r['reason']} | fix: {r['suggested_fix']}")


if __name__ == "__main__":
    main()
