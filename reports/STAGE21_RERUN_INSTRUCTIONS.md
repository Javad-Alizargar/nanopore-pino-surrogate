# Stage-21 verified rerun — handoff to Google Colab

## Why

The previous Stage-21 evidence was invalid:

- `results/_stage21_timing_cache.json` and `results/_stage21_bias_cache.json` were
  hand-authored placeholders with empty per-case rows.
- `results/_stage21_spline_cache.npz` (the basis of the “45 s for 2,700
  realizations” claim) did not contain the solved tables it claimed.
- The legacy `stage20_complete_records.npz/.csv` had all-zero standard errors,
  `B_delta_m_inf`, and coverage flags.

Those files were deleted, the loader now raises on missing diagnostics, and the
timing benchmark was genuinely rerun (`results/stage21_timing_benchmark_real.json`).
The inverse study still has to be rerun in full.

## Status: complete (675 realizations)

The verified rerun finished on Colab High-RAM (8 vCPU): **675 realizations**
(15 states × 3 protocols × 3 noise levels × 5 realizations), with
non-zero `B_se_ln*`, `B_delta_m_inf`, and coverage flags for all records.
Outputs are in `results/`:

- `stage21_inverse_records_real.npz`
- `stage21_inverse_summary_real.json`
- `stage21_inverse_metadata.json`
- `stage21_coverage_stats.json`
- `stage21_bias_diagnostic_real.json` (D1/D2/D3; local Mac run, identical to Colab)
- `table_s22_partB.md`

The same notebook can be re-run with `STAGE21_REPS=20` to extend the study to the
full 2,700 realizations; the checkpoint keys are (state, protocol, noise,
realization), so existing records can be reused by mapping.

## Run

1. Open `notebooks/stage21_inverse_colab.ipynb` in Colab.
2. Runtime → Change runtime type → CPU (Colab Pro / high-RAM with 4–8 vCPU is
   much faster; free tier has ~2 vCPU and will take ~11–13 h).
3. Run the cells in order:
   - mount Drive and locate the project,
   - `pip install onnxruntime`,
   - run/resume `scripts/run_stage21_colab.py`,
   - run `scripts/run_stage21_bias_real.py`,
   - run `scripts/summarize_stage21_real.py`,
   - download `results/stage21_colab_outputs.zip`.

If Colab disconnects, re-run the same cells. The driver skips the realizations
already in `results/checkpoints/inverse_records.pkl`.

## What lands in `results/`

| File | Contents |
|---|---|
| `stage21_inverse_records_real.npz` | 2,700 per-realization records with SEs, intervals, flags |
| `stage21_inverse_summary_real.json` | every summary recomputed from the records |
| `stage21_inverse_metadata.json` | configuration, code version, validation |
| `stage21_coverage_stats.json` | coverage by protocol/noise |
| `stage21_bias_diagnostic_real.json` | D1/D2/D3 rows and summary |
| `table_s22_partB.md` | markdown body for Table S22 part B |

## After the run

Send back `results/stage21_colab_outputs.zip` (or confirm the files are in the
synced Drive folder). The remaining steps are then:

1. Replace Table S22 part B with `table_s22_partB.md`; add the D1/D2/D3 rows to
   part C and restore the part-C caption.
2. Update every inverse number in the manuscript (abstract, §3.11, discussion,
   conclusion, highlights, cover letters) from
   `stage21_inverse_summary_real.json`.
3. Re-export both DOCX files, rebuild the packages, and rerun
   `scripts/audit_release_packages.py` to 37/37.
