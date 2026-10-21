# CPC test run

Self-contained verification case for the released PINO-Nanopore operator.

```bash
python scripts/run_cpc_test_case.py
```

Input: `sample_input.json`, the held-out interpolation benchmark state
`(V_app, C0, q_w, R) = (50 mV, 100 mol m^-3, -0.01, 5 nm)`.
The script executes `results/models/deeponet_reparam_seed42.onnx`, verifies
positive concentrations, the charge/salt-sum identities, reservoir boundary
values, and the stored summary in `expected_output.json`, then prints
`CPC TEST RUN PASSED`. Runtime is under one second on one CPU with only NumPy
and ONNX Runtime.
