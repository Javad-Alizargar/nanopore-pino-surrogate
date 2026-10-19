"""Build a verified, deduplicated literature database.

Sources: OpenAlex (key) + NCBI E-utilities (key) + Crossref (metadata verification).
Secrets are loaded in-process only and never printed. All HTTP responses are cached
under literature/cache/ for deterministic re-runs.

Outputs:
  manuscript/literature_db.json   canonical database
  manuscript/references.bib       BibTeX export
  literature/raw/literature_search_log.json
"""

import json
import os
import re
import sys
import time
import unicodedata
from datetime import date

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CACHE = os.path.join(PROJECT_ROOT, "literature", "cache")
RAW = os.path.join(PROJECT_ROOT, "literature", "raw")
MANU = os.path.join(PROJECT_ROOT, "manuscript")
PRIOR_FILE = os.path.join(RAW, "prior_da_pinn_references.txt")
TODAY = date.today().isoformat()
UA = "nanopore_pino_surrogate/0.1 (mailto:8javad@ntunhs.edu.tw)"
SLEEP = 0.12


def load_secret(rel):
    path = os.path.join(PROJECT_ROOT, rel)
    if not os.path.exists(path):
        return None
    raw = open(path, "r", encoding="utf-8", errors="ignore").read()
    sk = re.findall(r"sk-[A-Za-z0-9_\-]{10,}", raw)
    if sk:
        return max(sk, key=len)
    cand = [l.split("=", 1)[1] for l in raw.splitlines() if "=" in l]
    source = "\n".join(cand) if cand else raw
    toks = [t for t in re.findall(r"[A-Za-z0-9_\-\.]{16,}", source)
            if not re.fullmatch(r"[A-Z0-9_]+", t)]
    if not toks:
        toks = re.findall(r"[A-Za-z0-9_\-\.]{12,}", source)
    return max(toks, key=len) if toks else None


OPENALEX_KEY = load_secret("apis/OPENALEX.txt")
NCBI_KEY = load_secret("apis/NCBI.txt")


def _cache_path(key):
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", key)[:180]
    return os.path.join(CACHE, safe + ".json")


def get_json(url, params=None, cache_key=None, timeout=45):
    path = _cache_path(cache_key or url)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    headers = {"User-Agent": UA}
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    time.sleep(SLEEP)
    return data


def norm_title(t):
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def norm_doi(doi):
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi.rstrip(". ")


def openalex_by_doi(doi):
    url = f"https://api.openalex.org/works/doi:{doi}"
    params = {"api_key": OPENALEX_KEY} if OPENALEX_KEY else {}
    return get_json(url, params, cache_key=f"openalex_doi_{norm_doi(doi)}")


def crossref_by_doi(doi):
    url = f"https://api.crossref.org/works/{doi}"
    return get_json(url, {"mailto": "8javad@ntunhs.edu.tw"}, cache_key=f"crossref_doi_{norm_doi(doi)}")


def openalex_search(query, per_page=10):
    params = {
        "search": query,
        "per-page": per_page,
        "filter": "has_doi:true,type:article,from_publication_date:2018-01-01",
        "sort": "relevance_score:desc",
    }
    if OPENALEX_KEY:
        params["api_key"] = OPENALEX_KEY
    return get_json("https://api.openalex.org/works", params, cache_key=f"openalex_search_{norm_title(query)}")


def ncbi_search(term, retmax=8):
    params = {"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json"}
    if NCBI_KEY:
        params["api_key"] = NCBI_KEY
    es = get_json("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params,
                  cache_key=f"ncbi_esearch_{norm_title(term)}")
    ids = es.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    sp = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
    if NCBI_KEY:
        sp["api_key"] = NCBI_KEY
    summary = get_json("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", sp,
                       cache_key=f"ncbi_esummary_{'_'.join(ids)}")
    return [summary.get("result", {}).get(i, {}) for i in ids]


PRIOR_RE = re.compile(r"^\[(\d+)\]\s*(.*?)\s*doi:(10\.[^\s]+?)\.?\s*$")


def parse_prior():
    if not os.path.exists(PRIOR_FILE):
        return []
    out = []
    for line in open(PRIOR_FILE, encoding="utf-8"):
        m = PRIOR_RE.match(line.strip())
        if m:
            out.append({"n": int(m.group(1)), "raw": m.group(2).strip() + ".", "doi": norm_doi(m.group(3))})
    return out


TOPICS = {
    "fno_pino": [
        "Fourier neural operator for parametric partial differential equations",
        "physics-informed neural operator learning partial differential equations",
        "neural operator learning maps between function spaces",
        "geometry-aware Fourier neural operator boundary",
        "Fourier neural operator surrogate modeling",
    ],
    "deeponet": [
        "DeepONet learning nonlinear operators universal approximation theorem",
        "physics-informed DeepONet parametric partial differential equations",
        "operator learning parametric PDE surrogate DeepONet",
    ],
    "constraints_conditioning": [
        "physics-informed neural networks hard constraints boundary conditions",
        "exact imposition boundary conditions distance functions neural networks",
        "singular perturbation Poisson-Nernst-Planck boundary layer",
        "drift-diffusion semiconductor boundary layer asymptotic analysis",
    ],
    "nanopore_edl_design": [
        "electric double layer overlap nanochannel ion conductance",
        "surface charge nanopore ion selectivity ionic current simulation",
        "Poisson-Nernst-Planck nanopore numerical simulation",
        "inverse design nanopore ion transport machine learning",
        "nanopore sensing machine learning signal",
    ],
}

NCBI_QUERIES = [
    "nanopore Poisson-Nernst-Planck ion transport",
    "electric double layer overlap nanochannel conductance",
]


def entry_from_openalex(work, topic, origin):
    auth = [a.get("author", {}).get("display_name") for a in work.get("authorships", []) if a.get("author")]
    venue = None
    pl = work.get("primary_location") or {}
    src = pl.get("source") or {}
    venue = src.get("display_name")
    ids = work.get("ids", {})
    return {
        "title": work.get("title"),
        "authors": [a for a in auth if a],
        "year": work.get("publication_year"),
        "journal": venue,
        "doi": norm_doi(work.get("doi")),
        "pmid": (ids.get("pmid") or "").replace("https://pubmed.ncbi.nlm.nih.gov/", "").rstrip("/") or None,
        "url": work.get("doi") or ids.get("openalex"),
        "cited_by_count": work.get("cited_by_count"),
        "abstract_available": bool(work.get("abstract_inverted_index")),
        "topics": [topic],
        "origin": origin,
        "sources": ["openalex"],
    }


def verify_and_enrich(entry):
    doi = entry.get("doi")
    ver = {"crossref": False, "openalex": False, "title_match": None}
    if not doi:
        entry["verification"] = ver
        return entry
    try:
        cr = crossref_by_doi(doi)
        msg = cr.get("message", {})
        cr_title = (msg.get("title") or [""])[0]
        ver["crossref"] = True
        if not entry.get("title"):
            entry["title"] = cr_title
        if entry.get("title"):
            ver["title_match"] = norm_title(cr_title) == norm_title(entry["title"])
        if not entry.get("journal"):
            ct = msg.get("container-title") or []
            entry["journal"] = ct[0] if ct else entry.get("journal")
        if not entry.get("year"):
            parts = (msg.get("issued", {}) or {}).get("date-parts", [[None]])
            entry["year"] = parts[0][0] if parts and parts[0] else entry.get("year")
        if not entry.get("authors"):
            entry["authors"] = [f"{a.get('family','')}, {a.get('given','')}".strip(", ")
                                for a in msg.get("author", [])] or entry.get("authors")
        if "crossref" not in entry["sources"]:
            entry["sources"].append("crossref")
    except Exception:
        pass
    try:
        oa = openalex_by_doi(doi)
        ver["openalex"] = True
        if not entry.get("title"):
            entry["title"] = oa.get("title")
        if not entry.get("year"):
            entry["year"] = oa.get("publication_year")
        if not entry.get("journal"):
            pl = oa.get("primary_location") or {}
            entry["journal"] = (pl.get("source") or {}).get("display_name")
    except Exception:
        pass
    entry["verification"] = ver
    entry["retrieval_date"] = TODAY
    entry["url"] = f"https://doi.org/{doi}"
    return entry


def entry_from_ncbi(rec):
    doi = None
    for aid in rec.get("articleids", []):
        if aid.get("idtype") == "doi":
            doi = norm_doi(aid.get("value"))
    authors = [a.get("name") for a in rec.get("authors", []) if a.get("name")]
    return {
        "title": rec.get("title"),
        "authors": authors,
        "year": (rec.get("pubdate", "") or "")[:4] or None,
        "journal": rec.get("fulljournalname") or rec.get("source"),
        "doi": doi,
        "pmid": rec.get("uid"),
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{rec.get('uid')}/",
        "cited_by_count": None,
        "abstract_available": None,
        "topics": ["nanopore_edl_design"],
        "origin": "topic_ncbi",
        "sources": ["ncbi"],
    }


def score(e):
    v = e.get("verification", {})
    ok = 1 if (v.get("crossref") and v.get("title_match") is not False) else 0
    return (ok, e.get("cited_by_count") or 0, int(e.get("year") or 0))


QUOTAS = {"fno_pino": 8, "deeponet": 6, "constraints_conditioning": 6, "nanopore_edl_design": 6}


def select_canonical(entries):
    priors = [e for e in entries if e["origin"] == "prior_da_pinn"]
    topics = {}
    for e in entries:
        if e["origin"] == "prior_da_pinn":
            continue
        topics.setdefault(e["topics"][0], []).append(e)
    chosen = []
    for t, quota in QUOTAS.items():
        ranked = [e for e in topics.get(t, [])
                  if e.get("verification", {}).get("crossref") or e.get("verification", {}).get("openalex")]
        chosen.extend(ranked[:quota])
    return priors + chosen


def main():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(MANU, exist_ok=True)
    log = {"retrieval_date": TODAY, "queries": [], "errors": []}
    entries = []
    seen = set()

    priors = parse_prior()
    print(f"prior references parsed: {len(priors)}")
    for p in priors:
        e = {"title": None, "authors": [], "year": None, "journal": None,
             "doi": p["doi"], "pmid": None, "url": f"https://doi.org/{p['doi']}",
             "cited_by_count": None, "abstract_available": None,
             "topics": ["prior_da_pinn"], "origin": "prior_da_pinn", "sources": ["manuscript"]}
        try:
            e = verify_and_enrich(e)
        except Exception as exc:
            log["errors"].append({"doi": p["doi"], "error": str(exc)[:160]})
        e["raw_citation"] = p["raw"]
        entries.append(e)
        if p["doi"]:
            seen.add(p["doi"])

    for topic, queries in TOPICS.items():
        for q in queries:
            try:
                data = openalex_search(q, per_page=8)
            except Exception as exc:
                log["errors"].append({"query": q, "error": str(exc)[:160]})
                continue
            got = 0
            for w in data.get("results", []):
                doi = norm_doi(w.get("doi"))
                if not doi or doi in seen:
                    continue
                e = entry_from_openalex(w, topic, f"topic_{topic}")
                e = verify_and_enrich(e)
                entries.append(e)
                seen.add(doi)
                got += 1
            log["queries"].append({"source": "openalex", "topic": topic, "query": q, "new": got})

    for q in NCBI_QUERIES:
        try:
            recs = ncbi_search(q, retmax=8)
        except Exception as exc:
            log["errors"].append({"query": q, "error": str(exc)[:160]})
            continue
        got = 0
        for rec in recs:
            doi = norm_doi(rec.get("doi")) if any(a.get("idtype") == "doi" for a in rec.get("articleids", [])) else None
            key = doi or f"pmid:{rec.get('uid')}"
            if key in seen:
                continue
            e = entry_from_ncbi(rec)
            e = verify_and_enrich(e)
            entries.append(e)
            seen.add(key)
            got += 1
        log["queries"].append({"source": "ncbi", "topic": "nanopore_edl_design", "query": q, "new": got})

    candidates = entries
    selected = select_canonical(candidates)
    with open(os.path.join(RAW, "literature_candidates.json"), "w", encoding="utf-8") as fh:
        json.dump({"count": len(candidates), "entries": candidates}, fh, indent=2)

    for i, e in enumerate(selected, 1):
        e["id"] = f"ref{i:04d}"
    counts = {}
    for e in selected:
        counts[e["origin"]] = counts.get(e["origin"], 0) + 1
    verified = sum(1 for e in selected
                   if e.get("verification", {}).get("crossref") or e.get("verification", {}).get("openalex"))
    db = {
        "schema_version": "1.0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "retrieval_date": TODAY,
        "sources": ["OpenAlex", "NCBI E-utilities", "Crossref", "prior manuscript"],
        "candidate_pool_size": len(candidates),
        "counts": {"total": len(selected), "by_origin": counts, "crossref_verified": verified},
        "entries": selected,
    }
    with open(os.path.join(MANU, "literature_db.json"), "w", encoding="utf-8") as fh:
        json.dump(db, fh, indent=2)
    with open(os.path.join(RAW, "literature_search_log.json"), "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    write_bibtex(selected)
    print(json.dumps({"selected_total": len(selected), "candidate_pool": len(candidates),
                      "by_origin": counts, "crossref_verified": verified}, indent=2))


def bib_key(e, used):
    au = e.get("authors") or []
    last = "anon"
    if au:
        last = re.sub(r"[^A-Za-z]", "", au[0].split(",")[0].split(" ")[0]) or "anon"
    year = e.get("year") or "n.d."
    title = e.get("title") or ""
    word = next((re.sub(r"[^A-Za-z]", "", w) for w in title.split() if len(w) > 3), "ref")
    base = f"{last}{year}{word}"
    key = base
    n = 1
    while key in used:
        n += 1
        key = f"{base}{n}"
    used.add(key)
    return key


def write_bibtex(entries):
    used = set()
    lines = []
    for e in entries:
        key = bib_key(e, used)
        authors = " and ".join(e.get("authors") or []) or "Unknown"
        lines.append("@article{%s," % key)
        lines.append(f"  author  = {{{authors}}},")
        lines.append(f"  title   = {{{(e.get('title') or '').replace('{','').replace('}','')}}},")
        if e.get("journal"):
            lines.append(f"  journal = {{{e['journal']}}},")
        if e.get("year"):
            lines.append(f"  year    = {{{e['year']}}},")
        if e.get("doi"):
            lines.append(f"  doi     = {{{e['doi']}}},")
        if e.get("pmid"):
            lines.append(f"  pmid    = {{{e['pmid']}}},")
        if e.get("url"):
            lines.append(f"  url     = {{{e['url']}}},")
        lines.append(f"  note    = {{verified {e.get('retrieval_date', TODAY)}; origin {e.get('origin')}}},")
        lines.append("}")
        lines.append("")
    with open(os.path.join(MANU, "references.bib"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
