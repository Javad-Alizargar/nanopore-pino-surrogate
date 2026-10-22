#!/usr/bin/env python3
"""
Stage 21B Inverse Study and Bias Diagnostic - Google Colab Optimized Version
Runs with multiprocessing and checkpointing for long computations.
"""

import sys
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
import time
import json
import pickle
import multiprocessing as mp
from pathlib import Path

import numpy as np

# Add scripts to path
sys.path.insert(0, 'scripts')

import run_stage19_hybrid_solver_and_reversal as S19
import run_stage10_upgrades as S10

# Configuration
ONNX = "web_simulator/models/deeponet_reparam_seed42.onnx"
RESULTS_DIR = Path(os.environ.get("STAGE21_RESULTS_DIR", "results"))
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def initialize_globals():
    """Initialize global state for workers."""
    op = S10.OnnxOperator(ONNX)
    data = S10.load_npz(S10.DATA240)
    S19._OP = op
    return op, data

def _init_worker(onnx_path, rev_sur, state_data):
    """Initialize worker process with its own ONNX session and shared state."""
    op = S10.OnnxOperator(onnx_path)
    S19._OP = op
    S19._REV = rev_sur
    S19._STATE_DATA = state_data

def compute_truth_states(states):
    """Compute truth states on 41×121 grid with checkpointing."""
    checkpoint_file = CHECKPOINT_DIR / "truth_states.pkl"
    
    if checkpoint_file.exists():
        print(f"  Loading truth states from checkpoint...", flush=True)
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"  Computing {len(states)} truth states on 41×121 grid...", flush=True)
    t_start = time.perf_counter()
    truth = {}
    
    for sid, st in enumerate(states):
        print(f"    Truth state {sid+1}/{len(states)}...", flush=True)
        I41, V41 = S19.exact_protocol_values(st["q"], st["R"], st["C0"], "vrev", 41, 121)
        _, S41 = S19.exact_protocol_values(st["q"], st["R"], st["C0"], "dual", 41, 121)
        truth[sid] = {
            "I": I41, "Vrev": V41, "S": S41, 
            "q": st["q"], "R": st["R"], "C0": st["C0"],
            "sigma": st["sigma"], "band": st["band"]
        }
        
        # Checkpoint every 5 states
        if (sid + 1) % 5 == 0:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(truth, f)
            print(f"    ✓ Checkpoint saved at state {sid+1}", flush=True)
    
    # Final checkpoint
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(truth, f)
    
    elapsed = time.perf_counter() - t_start
    print(f"  ✓ Computed truth states in {elapsed:.1f}s", flush=True)
    return truth

def build_state_data(op, rev_sur, states):
    """Build candidate mesh predictions with checkpointing."""
    checkpoint_file = CHECKPOINT_DIR / "state_data.pkl"
    
    if checkpoint_file.exists():
        print(f"  Loading state data from checkpoint...", flush=True)
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"  Building candidate mesh predictions...", flush=True)
    qs, rs, cand = S19.candidate_mesh()
    state_data = {}
    
    for sid, st in enumerate(states):
        for protocol in ["single", "vrev", "dual"]:
            predI, predS, predV = S19.predict_candidates(op, qs, rs, st["C0"], protocol, rev_sur)
            state_data[(sid, protocol)] = (qs, rs, (predI, predS, predV))
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(state_data, f)
    
    print(f"  ✓ Built predictions for {len(states)} states × 3 protocols", flush=True)
    return state_data

def generate_tasks(states, truth, n_realizations=20):
    """Generate inverse tasks."""
    print(f"  Generating {n_realizations} realizations per condition...", flush=True)
    NOISE = [0.002, 0.01, 0.02]
    tasks = []
    
    for sid, st in enumerate(states):
        for protocol in ["single", "vrev", "dual"]:
            y_true = truth[sid]["I"].copy()
            if protocol == "vrev":
                y_true = np.r_[y_true, truth[sid]["Vrev"]]
            elif protocol == "dual":
                y_true = np.r_[y_true, truth[sid]["S"]]
            
            for noise in NOISE:
                for rep in range(n_realizations):
                    r = np.random.default_rng(
                        200100 + sid * 100000 + 
                        ["single", "vrev", "dual"].index(protocol) * 10000 + 
                        int(noise * 10000) * 100 + rep
                    )
                    sig = np.maximum(noise * np.abs(y_true), 1e-6)
                    if protocol == "vrev":
                        sig[-1] = max(noise * abs(y_true[-1]), 0.05)
                    elif protocol == "dual":
                        sig[-1] = noise * max(abs(y_true[-1]), 0.01)
                    
                    y_obs = y_true + sig * r.standard_normal(y_true.size)
                    tasks.append((
                        sid, protocol, noise, rep, y_obs, sig,
                        {"q": st["q"], "R": st["R"], "C0": st["C0"]}
                    ))
    
    print(f"  ✓ Generated {len(tasks)} tasks", flush=True)
    return tasks

def run_inverse_tasks_parallel(tasks, op, rev_sur, state_data, n_workers=None):
    """Run inverse tasks with multiprocessing and checkpointing."""
    if n_workers is None:
        n_workers = int(os.environ.get("STAGE21_WORKERS", "0")) or max(1, (os.cpu_count() or 2))
    
    print(f"  Running {len(tasks)} tasks with {n_workers} workers...", flush=True)
    
    checkpoint_file = CHECKPOINT_DIR / "inverse_records.pkl"
    progress_file = CHECKPOINT_DIR / "inverse_progress.json"
    
    # Load checkpoint if exists
    completed_indices = set()
    records = []
    
    if checkpoint_file.exists():
        print(f"  Loading checkpoint...", flush=True)
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            records = checkpoint_data['records']
            completed_indices = set(checkpoint_data['completed_indices'])
        print(f"  ✓ Resumed from {len(records)}/{len(tasks)} tasks", flush=True)
    
    # Filter out completed tasks
    remaining_tasks = [(i, task) for i, task in enumerate(tasks) if i not in completed_indices]
    
    if not remaining_tasks:
        print(f"  ✓ All tasks already completed", flush=True)
        return records
    
    t_start = time.perf_counter()
    
    # Use multiprocessing pool with initializer
    with mp.Pool(processes=n_workers, initializer=_init_worker, 
                 initargs=(ONNX, rev_sur, state_data)) as pool:
        # Process in batches for checkpointing
        batch_size = 50
        for batch_start in range(0, len(remaining_tasks), batch_size):
            batch = remaining_tasks[batch_start:batch_start + batch_size]
            batch_indices = [i for i, _ in batch]
            batch_tasks = [task for _, task in batch]
            
            # Run batch
            batch_results = pool.map(S19._inverse_task, batch_tasks)
            
            # Add to records
            for idx, result in zip(batch_indices, batch_results):
                records.append(result)
                completed_indices.add(idx)
            
            # Save checkpoint
            checkpoint_data = {
                'records': records,
                'completed_indices': list(completed_indices)
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Progress update
            elapsed = time.perf_counter() - t_start
            completed = len(records)
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (len(tasks) - completed) / rate if rate > 0 else 0
            
            print(f"    Progress: {completed}/{len(tasks)} tasks "
                  f"({elapsed:.0f}s elapsed, ~{remaining/60:.1f}min remaining)", flush=True)
    
    elapsed = time.perf_counter() - t_start
    print(f"  ✓ Completed {len(records)} tasks in {elapsed:.1f}s "
          f"({elapsed/len(records):.2f}s per task)", flush=True)
    
    return records

def validate_records(records):
    """Validate that diagnostics are properly computed."""
    print("\n" + "="*60)
    print("VALIDATION OF DIAGNOSTICS")
    print("="*60)
    
    se_fields = ["B_se_lnR", "B_se_lnsigma", "B_se_lnq", 
                 "C_se_lnR", "C_se_lnsigma", "C_se_lnq"]
    delta_fields = ["B_delta_m_inf", "C_delta_m_inf"]
    
    all_pass = True
    
    for field in se_fields:
        values = [r[field] for r in records]
        nonzero = sum(1 for v in values if v > 0)
        if nonzero == 0:
            print(f"  ✗ FAIL: {field} is all zeros", flush=True)
            all_pass = False
        else:
            print(f"  ✓ PASS: {field} has {nonzero}/{len(values)} non-zero values "
                  f"(mean={np.mean(values):.6f})", flush=True)
    
    for field in delta_fields:
        values = [r[field] for r in records]
        nonzero = sum(1 for v in values if v > 0)
        if nonzero == 0:
            print(f"  ✗ FAIL: {field} is all zeros", flush=True)
            all_pass = False
        else:
            print(f"  ✓ PASS: {field} has {nonzero}/{len(values)} non-zero values", flush=True)
    
    return all_pass

def compute_coverage_statistics(records):
    """Compute coverage statistics from real data."""
    print("\n" + "="*60)
    print("COVERAGE STATISTICS (from real data)")
    print("="*60)
    
    NOISE = [0.002, 0.01, 0.02]
    stats = {}
    
    for protocol in ["single", "vrev", "dual"]:
        for noise in NOISE:
            subset = [r for r in records 
                     if r["protocol"] == protocol and abs(r["noise"] - noise) < 1e-9]
            
            if subset:
                q_cover = sum(1 for r in subset if r["B_q_cover"]) / len(subset)
                R_cover = sum(1 for r in subset if r["B_R_cover"]) / len(subset)
                sigma_cover = sum(1 for r in subset if r["B_sigma_cover"]) / len(subset)
                
                key = f"{protocol}_{noise}"
                stats[key] = {
                    "n": len(subset),
                    "q_cover": q_cover,
                    "R_cover": R_cover,
                    "sigma_cover": sigma_cover,
                    "median_q_rel": float(np.median([r["B_q_rel"] for r in subset])),
                    "median_R_rel": float(np.median([r["B_R_rel"] for r in subset])),
                    "median_sigma_rel": float(np.median([r["B_sigma_rel"] for r in subset]))
                }
                
                print(f"  {protocol:8s} @ {noise*100:.1f}% noise: "
                      f"q={q_cover:.1%}, R={R_cover:.1%}, σ={sigma_cover:.1%} "
                      f"(n={len(subset)})", flush=True)
    
    return stats

def save_final_results(records, coverage_stats, metadata):
    """Save final results with proper structure."""
    print("\n" + "="*60)
    print("SAVING FINAL RESULTS")
    print("="*60)
    
    # Save records to NPZ
    npz_path = RESULTS_DIR / "stage21_inverse_records_real.npz"
    arrays = {}
    
    for key in records[0].keys():
        if key in ["y_obs", "sigmas"]:
            continue
        try:
            arrays[key] = np.array([r[key] for r in records])
        except:
            pass
    
    np.savez_compressed(npz_path, **arrays)
    print(f"  ✓ Saved {len(records)} records to {npz_path}", flush=True)
    
    # Save coverage stats
    coverage_path = RESULTS_DIR / "stage21_coverage_stats.json"
    with open(coverage_path, 'w') as f:
        json.dump(coverage_stats, f, indent=2)
    print(f"  ✓ Saved coverage statistics to {coverage_path}", flush=True)
    
    # Save metadata
    metadata_path = RESULTS_DIR / "stage21_inverse_metadata.json"
    metadata.update({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_records": len(records),
        "validation": {
            "all_diagnostics_populated": validate_records(records)
        }
    })
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata to {metadata_path}", flush=True)

def main():
    """Main execution."""
    print("="*60)
    print("STAGE 21B INVERSE STUDY - COLAB OPTIMIZED")
    print("="*60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Workers: {os.cpu_count()} CPUs available", flush=True)
    
    # Initialize
    t_total_start = time.perf_counter()
    op, data = initialize_globals()
    
    # Build states
    print("\n[1/6] Building 15 truth states...", flush=True)
    rng = np.random.default_rng(200020)
    bands = [("weak", 0.04, 0.15), ("transitional", 0.15, 0.60), ("strong", 0.60, 1.50)]
    states = []
    for band, lo, hi in bands:
        for _ in range(5):
            R = float(rng.uniform(3.0, 7.0))
            C0 = float(10 ** rng.uniform(np.log10(15.0), np.log10(150.0)))
            sigma = -float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
            q = float(S19.q_from_sigma(sigma, R))
            states.append({"band": band, "R": R, "C0": C0, "sigma": sigma, "q": q})
    
    # Compute truth states
    print("\n[2/6] Computing truth states...", flush=True)
    truth = compute_truth_states(states)
    
    # Fit calibrations
    print("\n[3/6] Fitting calibrations...", flush=True)
    cal = S19.fit_calibration(op, data)
    rev_sur, n_rev = S19.fit_reversal_surrogate(op, data)
    S19._REV = rev_sur
    
    # Build state data
    print("\n[4/6] Building state data...", flush=True)
    state_data = build_state_data(op, rev_sur, states)
    S19._STATE_DATA = state_data
    
    # Generate tasks
    print("\n[5/6] Generating tasks...", flush=True)
    tasks = generate_tasks(states, truth, n_realizations=20)
    
    # Run inverse study
    print("\n[6/6] Running inverse study...", flush=True)
    records = run_inverse_tasks_parallel(tasks, op, rev_sur, state_data)
    
    # Validate
    validation_pass = validate_records(records)
    
    # Compute coverage
    coverage_stats = compute_coverage_statistics(records)
    
    # Save results
    metadata = {
        "n_states": len(states),
        "n_realizations_per_condition": 20,
        "total_tasks": len(tasks),
        "elapsed_time_s": time.perf_counter() - t_total_start
    }
    save_final_results(records, coverage_stats, metadata)
    
    total_time = time.perf_counter() - t_total_start
    print("\n" + "="*60)
    print("COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)", flush=True)
    print(f"Records: {len(records)}", flush=True)
    print(f"Validation: {'✓ PASS' if validation_pass else '✗ FAIL'}", flush=True)

if __name__ == "__main__":
    # Set multiprocessing start method for Colab compatibility
    mp.set_start_method('spawn', force=True)
    main()
