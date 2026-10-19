"""TASK 6B.1 - Independent cross-provider scientific peer review.

Reviewer 1: OpenAI gpt-4.1-mini (methodological / mathematical rigor).
Reviewer 2: DeepSeek deepseek-v4-pro (software / reproducibility / physical
electrochemistry).

Both requests are routed through orchestration/budget_ledger.py with a shared
task allowance of $0.15. Credential values are never printed or logged.

    .venv/bin/python scripts/run_independent_review.py [--dry-run]
"""

import json
import os
import re
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "orchestration"))
from budget_ledger import BudgetLedger, Denied  # noqa: E402

CFG = json.load(open(os.path.join(ROOT, "config", "project_config.json"), encoding="utf-8"))
CRED = CFG["credential_file_map"]
OUT = os.path.join(ROOT, "results", "independent_review.json")
TASK_ID = "stage6b_independent_review"
TASK_ALLOWANCE_USD = 0.15
MAX_OUT = 3500

REVIEWERS = [
    {
        "id": "reviewer1_openai",
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "role": "Reviewer 1 - JCP/CPC Methodological & Mathematical Rigor Reviewer",
    },
    {
        "id": "reviewer2_deepseek",
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "role": "Reviewer 2 - CPC Software, Reproducibility & Physical Electrochemistry Reviewer",
    },
]

CORE_INSTRUCTIONS = """You are an independent, critical peer reviewer for a manuscript
submitted to the Journal of Computational Physics / Computer Physics Communications.
You have the full manuscript, the supplementary materials, and a claim-evidence map
below. Do NOT rewrite the paper. Audit it against the following checklist:

1. Mathematical clarity and completeness of the exact-boundary and charge-balanced
   hyperbolic ansatz (s = z(1-z)G_s, rho = z(1-z)G_rho, c_± = (sqrt(s^2+rho^2) ± rho)/2),
   the reservoir Dirichlet construction, and the finite-volume / Scharfetter-Gummel
   operator and physics loss. Flag undefined symbols, sign errors, missing factors,
   or claims that the equations do not support.
2. Readability and typography of numbers in the narrative. Raw 15-17 digit Python
   floats (e.g. 0.0045737918517061785 or 50.59363783302251 s) are inappropriate in
   prose; they should be rounded to 3-4 significant figures (e.g. 0.00457, 0.46%,
   50.6 s, 0.072 ms) while machine-precision values stay in JSON/CSV tables. List
   every location you consider a typography problem, and say whether exact values
   should remain in a table/archive.
3. Completeness of the CPC Program Summary block, table headers/units, figure
   captions, and the physical interpretation of the inverse-problem identifiability
   finding (radius identifiable; surface charge is a flat direction for
   conductance-only inversion). Flag over-claims, missing caveats, or unsupported
   causal language.
4. Internal inconsistencies between abstract, results, tables, claims, and the
   supplement; missing definitions; unsupported comparisons; reproducibility gaps.

Output ONLY a JSON object (no markdown fence) with this schema:
{
  "reviewer": "<your role>",
  "recommendation": "accept|minor revision|major revision|reject",
  "summary": "<=120 words",
  "issues": [
    {
      "id": "R1-1",
      "severity": "major|minor|editorial",
      "location": "section/table/figure/line hint",
      "evidence": "quote or precise description",
      "problem": "what is wrong or risky",
      "fix": "concrete, minimal fix"
    }
  ],
  "strengths": ["<=4 short items"]
}
Report at most 12 issues, ordered by severity. Be specific and cite quoted text.
If a concern is only stylistic, mark it editorial. Do not invent numbers.
"""


def load_secret(rel):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        return None
    raw = open(path, encoding="utf-8", errors="ignore").read()
    sk = re.findall(r"sk-[A-Za-z0-9_\-]{10,}", raw)
    if sk:
        return max(sk, key=len)
    toks = re.findall(r"[A-Za-z0-9_\-\.]{16,}", raw)
    toks = [t for t in toks if not re.fullmatch(r"[A-Z0-9_]+", t)]
    toks = toks or re.findall(r"[A-Za-z0-9_\-\.]{12,}", raw)
    return max(toks, key=len) if toks else None


def read(rel):
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


def build_prompt(role, manuscript, supplement, claims):
    return (
        CORE_INSTRUCTIONS
        + "\n\nYOUR ASSIGNED ROLE: " + role
        + "\n\n===== MANUSCRIPT (manuscript/Manuscript.md) =====\n" + manuscript
        + "\n\n===== SUPPLEMENT (manuscript/Supplementary_Materials.md) =====\n" + supplement
        + "\n\n===== CLAIM-EVIDENCE MAP (manuscript/claim_evidence.csv) =====\n" + claims
    )


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    global MAX_OUT
    dry = "--dry-run" in sys.argv
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    if "--max-out" in sys.argv:
        MAX_OUT = int(sys.argv[sys.argv.index("--max-out") + 1])
    model_override = None
    if "--model" in sys.argv:
        model_override = sys.argv[sys.argv.index("--model") + 1]
    manuscript = read(os.path.join("manuscript", "Manuscript.md"))
    supplement = read(os.path.join("manuscript", "Supplementary_Materials.md"))
    claims = read(os.path.join("manuscript", "claim_evidence.csv"))

    led = BudgetLedger()
    if os.path.exists(OUT) and only and not dry:
        report = json.load(open(OUT, encoding="utf-8"))
        report["task_allowance_usd"] = TASK_ALLOWANCE_USD
    else:
        report = {"task_id": TASK_ID, "task_allowance_usd": TASK_ALLOWANCE_USD,
                  "dry_run": dry, "reviewers": {}, "total_spend_usd": 0.0}

    for rev in REVIEWERS:
        if only and rev["id"] != only:
            continue
        if model_override:
            rev = dict(rev, model=model_override)
        prompt = build_prompt(rev["role"], manuscript, supplement, claims)
        est_in = int(len(prompt) / 3.2) + 200
        reserved_est, detail = led.compute_reservation_cost(
            rev["provider"], rev["model"], est_in, MAX_OUT)
        entry = {"provider": rev["provider"], "model": rev["model"],
                 "role": rev["role"], "est_input_tokens": est_in,
                 "max_output_tokens": MAX_OUT,
                 "estimated_reserved_cost_usd": reserved_est}
        if dry:
            entry["status"] = "dry_run"
            report["reviewers"][rev["id"]] = entry
            print(f"[review] {rev['id']}: est_reserved=${reserved_est:.5f} "
                  f"in={est_in} out={MAX_OUT}")
            continue

        key = load_secret(CRED[rev["provider"]])
        if not key:
            entry["status"] = "no_credential"
            report["reviewers"][rev["id"]] = entry
            print(f"[review] {rev['id']}: credential missing; skipped")
            continue

        try:
            res = led.admit(provider=rev["provider"], model=rev["model"],
                            est_input_tokens=est_in, max_output_tokens=MAX_OUT,
                            task_id=TASK_ID, task_budget_usd=TASK_ALLOWANCE_USD)
        except Denied as exc:
            entry["status"] = "denied"
            entry["reason"] = exc.reason
            entry["detail"] = exc.detail
            report["reviewers"][rev["id"]] = entry
            print(f"[review] {rev['id']}: DENIED {exc.reason}")
            continue

        rid = res["reservation_id"]
        entry["reserved_cost_usd"] = res["reserved_cost_usd"]
        base = led.models["providers"][rev["provider"]]["endpoint"].rstrip("/")
        body = {
            "model": rev["model"],
            "temperature": 0,
            "max_tokens": MAX_OUT,
            "messages": [{"role": "user", "content": prompt}],
        }
        if rev["provider"] == "deepseek":
            body["thinking"] = {"type": "disabled"}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            r = requests.post(base + "/chat/completions", headers=headers,
                              json=body, timeout=300)
            if r.status_code == 200:
                j = r.json()
                usage = j.get("usage", {}) or {}
                rec = led.reconcile(rid, {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                        "cached_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens", 0),
                })
                txt = j["choices"][0]["message"]["content"]
                parsed = extract_json(txt)
                entry.update({"status": "ok",
                              "calc_cost_usd": rec["calc_cost_usd"],
                              "usage": usage,
                              "raw_response": txt,
                              "parsed": parsed})
                report["total_spend_usd"] = round(
                    report["total_spend_usd"] + rec["calc_cost_usd"], 8)
                n = len(parsed.get("issues", [])) if parsed else "?"
                print(f"[review] {rev['id']}: ok, issues={n}, "
                      f"cost=${rec['calc_cost_usd']:.5f}")
            else:
                led.mark_timeout(rid, reason=f"http_{r.status_code}")
                entry.update({"status": "http_error", "http_status": r.status_code,
                              "body_snippet": r.text[:300]})
                print(f"[review] {rev['id']}: HTTP {r.status_code}")
        except requests.RequestException as exc:
            led.mark_timeout(rid, reason="transport_error")
            entry.update({"status": "transport_error", "error": type(exc).__name__})
            print(f"[review] {rev['id']}: transport error {type(exc).__name__}")
        report["reviewers"][rev["id"]] = entry

    report["ledger_status_after"] = led.status()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[review] total spend ${report['total_spend_usd']} -> "
          f"{os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
