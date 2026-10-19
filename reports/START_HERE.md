# START HERE — Launcher & Resume Procedure

Project: `nanopore_pino_surrogate` · Stage 1 complete · Date: 2026-10-18

## 0. One-line orientation
We are upgrading the fixed-state DA-PINN nanopore study into a **parameter-conditioned physics-informed neural operator** for 2D axisymmetric PNP transport over $\boldsymbol{\mu}=(V_{\text{app}}, C_0, q_w, R)$. Stage 1 (setup/accounting/brief) is done; no training has run yet.

## 1. Paths
```
PROJECT_ROOT=/Users/javadalizargar/Google drive MAC/PINN to a Neural Operator
config/        project_config.json, models.json, pricing.json
orchestration/ budget_ledger.py, test_budget_ledger.py
scripts/       smoke_connectivity.py
reports/       CAPABILITY_REPORT.md, PROJECT_BRIEF.md, START_HERE.md, STATUS.md, DECISIONS.md,
               NEXT_ACTIONS.md, REPORT_FOR_CHATGPT.md
logs/          budget_ledger.sqlite, environment_stage1.json, smoke_connectivity.json
apis/          credentials (git-ignored; never print contents)
```

## 2. Environment activation (to do in Stage 2)
A project virtual environment is **not yet created**. Create and populate it before training:
```
cd "/Users/javadalizargar/Google drive MAC/PINN to a Neural Operator"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy scipy pandas torch matplotlib python-docx \
    onnx onnxruntime scikit-learn requests
# choose ONE UI framework later: gradio OR streamlit
```
`.venv/` is git-ignored.

## 3. Accounting — always go through the ledger
All scripted paid calls MUST pass admission first:
```python
from budget_ledger import BudgetLedger, Denied
led = BudgetLedger()
res = led.admit(provider="deepseek", model="deepseek-flash",
                est_input_tokens=2000, max_output_tokens=2000, task_id="task-001")
# ... make the provider call ...
led.reconcile(res["reservation_id"], usage)
# on timeout / lost stream / crash: led.mark_timeout(res["reservation_id"])
```
CLI:
```
python3 orchestration/budget_ledger.py status
python3 orchestration/budget_ledger.py recover --max-age-seconds 900
python3 orchestration/budget_ledger.py events
```
Dry-run test suite (zero paid calls):
```
cd orchestration && python3 -m unittest -v test_budget_ledger
```

## 4. Smoke test (costs a few hundredths of a cent)
```
python3 scripts/smoke_connectivity.py            # all providers + metadata APIs
python3 scripts/smoke_connectivity.py openai     # single provider only
```

## 5. Resume procedure (after interruption)
1. Read `reports/STATUS.md`, `reports/DECISIONS.md`, `reports/NEXT_ACTIONS.md`.
2. `python3 orchestration/budget_ledger.py recover` — reconcile/stop any outstanding reservations before launching new work.
3. `python3 orchestration/budget_ledger.py status` — check spend and remaining budget.
4. Continue from the first unchecked item in `reports/NEXT_ACTIONS.md`.
5. Do **not** repeat completed searches/analyses unless inputs changed.

## 6. Restart / worker launch status
- No worker pool or local gateway is installed. Stage 1 established only: config, directory tree, ledger + admission control, and a verified launcher. Workers (if used) will be independent `python3` child processes in isolated `workers/<task_id>/` directories; their paid calls must also pass the ledger. Do not claim workers exist until real PIDs and bounded task outputs are recorded.

## 7. Guardrails
- Never print/echo/log credential values. Only report presence + byte length.
- No GitHub push, no public release, no deployment, no submission without explicit user command.
- Respect `STOP`/`STOP_REQUESTED`.
