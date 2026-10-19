"""One tiny live connectivity smoke test per provider + read-only metadata APIs.

Security: credential values are loaded into memory only and are never printed,
logged, or written. Any output string is scrubbed of known secret values before
printing. Paid chat calls are capped at max_tokens=16 and routed through the
budget ledger admission controller.
"""

import json
import os
import re
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "orchestration"))

from budget_ledger import BudgetLedger, Denied  # noqa: E402

APIS_DIR = os.path.join(PROJECT_ROOT, "apis")
CONFIG = json.load(open(os.path.join(PROJECT_ROOT, "config", "project_config.json"), encoding="utf-8"))
CRED = CONFIG["credential_file_map"]
MAX_TOKENS = 16


def load_secret(rel_path):
    path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()
    sk = re.findall(r"sk-[A-Za-z0-9_\-]{10,}", raw)
    if sk:
        return max(sk, key=len)
    candidates = []
    if "=" in raw:
        for line in raw.splitlines():
            if "=" in line:
                candidates.append(line.split("=", 1)[1])
    source = "\n".join(candidates) if candidates else raw
    tokens = re.findall(r"[A-Za-z0-9_\-\.]{16,}", source)
    tokens = [t for t in tokens if not re.fullmatch(r"[A-Z0-9_]+", t)]
    if not tokens:
        tokens = re.findall(r"[A-Za-z0-9_\-\.]{12,}", source)
    if not tokens:
        return None
    return max(tokens, key=len)


def scrub(text, secrets):
    out = str(text)
    for s in secrets:
        if s:
            out = out.replace(s, "[REDACTED]")
    return out


def _endpoint(base, path):
    base = base.rstrip("/")
    if base.endswith("/v1"):
        return base + path
    return base + "/v1" + path


def api_get_models(base, key, timeout=30):
    headers = {"Authorization": f"Bearer {key}"}
    r = requests.get(_endpoint(base, "/models"), headers=headers, timeout=timeout)
    ids = []
    if r.status_code == 200:
        try:
            ids = [m.get("id") for m in r.json().get("data", [])]
        except Exception:
            ids = []
    return r.status_code, ids


def chat_call(base, key, model, timeout=60):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    r = requests.post(_endpoint(base, "/chat/completions"), headers=headers, json=body, timeout=timeout)
    usage = None
    if r.status_code == 200:
        try:
            usage = r.json().get("usage")
        except Exception:
            usage = None
    return r.status_code, usage


def provider_smoke(led, provider, base, model_pref, catalog_models, secrets, results):
    key = load_secret(CRED.get(provider))
    if not key:
        results[provider] = {"present": False, "status": "skipped_no_key"}
        return
    entry = {"present": True}
    try:
        status, ids = api_get_models(base, key)
        entry["models_list_status"] = status
        available = [m for m in catalog_models if m in ids]
        if not ids:
            entry["status"] = "models_list_unavailable"
            results[provider] = entry
            return
        model = model_pref if model_pref in ids else (available[0] if available else None)
        if model is None:
            entry["status"] = "no_priced_model_available"
            entry["priced_available"] = available
            results[provider] = entry
            return
        entry["model"] = model
        reservation = led.admit(
            provider=provider, model=model, est_input_tokens=64,
            max_output_tokens=MAX_TOKENS, task_id="stage1_smoke", essential=True,
        )
        rid = reservation["reservation_id"]
        entry["reserved_cost_usd"] = reservation["reserved_cost_usd"]
        try:
            status, usage = chat_call(base, key, model)
            entry["chat_status"] = status
            if status == 200 and usage:
                rec = led.reconcile(rid, {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    if isinstance(usage.get("prompt_tokens_details"), dict) else 0,
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                    if isinstance(usage.get("completion_tokens_details"), dict) else 0,
                })
                entry["status"] = "ok"
                entry["usage"] = usage
                entry["calc_cost_usd"] = rec["calc_cost_usd"]
                entry["billing_tier"] = rec["billing_tier"]
            else:
                led.mark_timeout(rid, reason=f"http_{status}")
                entry["status"] = "http_error_retained_unresolved"
                entry["note"] = "non-2xx response; reservation conservatively retained"
        except requests.RequestException as exc:
            led.mark_timeout(rid, reason="transport_error")
            entry["status"] = "transport_error"
            entry["error"] = scrub(exc, secrets)
    except Denied as exc:
        entry["status"] = "denied_by_ledger"
        entry["reason"] = exc.reason
    results[provider] = entry


def openalex_smoke(secrets, results):
    key = load_secret(CRED.get("openalex"))
    if not key:
        results["openalex"] = {"present": False, "status": "skipped_no_key"}
        return
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={"per_page": 1, "api_key": key, "search": "Poisson Nernst Planck nanopore"},
            timeout=30,
        )
        entry = {"present": True, "status": "ok" if r.status_code == 200 else "http_error",
                 "http_status": r.status_code}
        if r.status_code == 200:
            entry["count"] = r.json().get("meta", {}).get("count")
        else:
            entry["body"] = scrub(r.text[:300], secrets)
        results["openalex"] = entry
    except requests.RequestException as exc:
        results["openalex"] = {"present": True, "status": "transport_error", "error": scrub(exc, secrets)}


def ncbi_smoke(secrets, results):
    key = load_secret(CRED.get("ncbi"))
    if not key:
        results["ncbi"] = {"present": False, "status": "skipped_no_key"}
        return
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": "nanopore Poisson-Nernst-Planck", "retmax": 1,
                    "retmode": "json", "api_key": key},
            timeout=30,
        )
        entry = {"present": True, "status": "ok" if r.status_code == 200 else "http_error",
                 "http_status": r.status_code}
        if r.status_code == 200:
            entry["count"] = r.json().get("esearchresult", {}).get("count")
        else:
            entry["body"] = scrub(r.text[:300], secrets)
        results["ncbi"] = entry
    except requests.RequestException as exc:
        results["ncbi"] = {"present": True, "status": "transport_error", "error": scrub(exc, secrets)}


def main():
    led = BudgetLedger()
    secrets = [
        load_secret(CRED.get("openai")),
        load_secret(CRED.get("deepseek")),
        load_secret(CRED.get("openalex")),
        load_secret(CRED.get("ncbi")),
        load_secret(CRED.get("github")),
    ]
    secrets = [s for s in secrets if s]
    results = {}
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    do = lambda name: (not only) or (name in only)  # noqa: E731
    if do("deepseek"):
        provider_smoke(led, "deepseek", "https://api.deepseek.com", "deepseek-flash",
                       ["deepseek-flash", "deepseek-v4-pro"], secrets, results)
    if do("openai"):
        provider_smoke(led, "openai", "https://api.openai.com/v1", "gpt-4o-mini",
                       ["gpt-4o-mini", "gpt-4.1-mini"], secrets, results)
    if do("openalex"):
        openalex_smoke(secrets, results)
    if do("ncbi"):
        ncbi_smoke(secrets, results)
    results["ledger_status_after"] = led.status()

    out_path = os.path.join(PROJECT_ROOT, "logs", "smoke_connectivity.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
