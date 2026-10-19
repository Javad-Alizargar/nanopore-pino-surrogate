"""TASK 5B.2 — budgeted gpt-4o-mini vision QA of Figures 1-6.

Sends each rendered PNG (downscaled to <=1024 px wide) to gpt-4o-mini with a
strict-JSON rubric asking for any remaining text overlap, clipping or overflow.
Every call is routed through the budget-ledger admission controller with a hard
task allowance of $0.03.  Credential values are never printed or logged.

    .venv/bin/python scripts/vision_qa_figures.py [fig1 ...]
"""

import base64
import io
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
FIGDIR = os.path.join(ROOT, "manuscript", "figures")
OUT = os.path.join(ROOT, "results", "figure_data", "vision_qa_report.json")
MODEL = "gpt-4o-mini"
TASK_ID = "stage5b_vision_qa"
TASK_ALLOWANCE_USD = 0.03
MAX_OUT = 400
BASE = "https://api.openai.com/v1"

FIGURES = [
    "fig1_workflow_and_architecture",
    "fig2_benchmark_and_field_reconstruction",
    "fig3_h3_charge_reparameterization_ablation",
    "fig4_h1_h2_interp_vs_ood_regimes",
    "fig5_parametric_physics_sweeps",
    "fig6_h4_timing_and_amortization_breakeven",
]

PROMPT = (
    "You are a scientific-figure layout reviewer. Inspect this figure image and "
    "report ONLY layout defects, using this exact JSON schema and nothing else:\n"
    '{"pass": true|false, '
    '"text_overlaps": ["..."], '
    '"clipped_or_overflowing_text": ["..."], '
    '"other_issues": ["..."], '
    '"notes": "one short sentence"}\n'
    "Rules: pass=true only if NO text overlaps another text and NO text is clipped "
    "or spills outside its box/axes/figure. Overlapping data points or gridlines are "
    "fine. Keep each list <=4 items; use short locations like 'panel B top-left'. "
    "Output JSON only."
)


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


def data_url(png_path, max_w=1024):
    from PIL import Image
    im = Image.open(png_path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), buf.tell()


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    which = [a for a in sys.argv[1:] if not a.startswith("-")] or FIGURES
    key = load_secret(CRED["openai"])
    if not key:
        print("no openai credential present; skipping vision QA")
        return
    led = BudgetLedger()
    report = {"model": MODEL, "task_allowance_usd": TASK_ALLOWANCE_USD,
              "figures": {}, "total_spend_usd": 0.0}
    for name in which:
        png = os.path.join(FIGDIR, name + ".png")
        url, nbytes = data_url(png)
        est_in = 800 + len(PROMPT) // 4
        entry = {"bytes_sent": nbytes}
        try:
            res = led.admit(provider="openai", model=MODEL, est_input_tokens=est_in,
                            max_output_tokens=MAX_OUT, task_id=TASK_ID,
                            task_budget_usd=TASK_ALLOWANCE_USD)
        except Denied as exc:
            entry["status"] = "denied"
            entry["reason"] = exc.reason
            report["figures"][name] = entry
            continue
        rid = res["reservation_id"]
        entry["reserved_cost_usd"] = res["reserved_cost_usd"]
        body = {
            "model": MODEL, "temperature": 0, "max_tokens": MAX_OUT,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
            ]}],
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            r = requests.post(BASE + "/chat/completions", headers=headers,
                              json=body, timeout=120)
            if r.status_code == 200:
                j = r.json()
                usage = j.get("usage", {}) or {}
                rec = led.reconcile(rid, {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                })
                txt = j["choices"][0]["message"]["content"]
                parsed = extract_json(txt)
                entry.update({"status": "ok", "calc_cost_usd": rec["calc_cost_usd"],
                              "verdict": parsed if parsed else {"raw": txt[:400]}})
                report["total_spend_usd"] = round(
                    report["total_spend_usd"] + rec["calc_cost_usd"], 8)
            else:
                led.mark_timeout(rid, reason=f"http_{r.status_code}")
                entry.update({"status": "http_error", "http_status": r.status_code})
        except requests.RequestException as exc:
            led.mark_timeout(rid, reason="transport_error")
            entry["status"] = "transport_error"
            entry["error"] = type(exc).__name__
        report["figures"][name] = entry
        v = entry.get("verdict", {})
        print(f"[vision] {name}: status={entry.get('status')} pass={v.get('pass')} "
              f"cost=${entry.get('calc_cost_usd', 0)}")

    report["ledger_status_after"] = led.status()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[vision] total spend ${report['total_spend_usd']} -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
