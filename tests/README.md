# Tests

- `test_onnx_parity.py` checks that both ONNX models load and that an
  onnxruntime forward pass runs on a random normalized input.
- `run_tests.sh` runs the package smoke tests. Some tests need the dataset
  under `data/processed/` and the models under `results/models/`.
