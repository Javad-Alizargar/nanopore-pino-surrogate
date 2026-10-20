"""Regenerate corrected references, tolerance/grid studies and paired PINN runs."""
import argparse
from dataclasses import replace, asdict
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pnp_da_pinn.baseline import PnpSolution
from pnp_da_pinn.config import load_config
from pnp_da_pinn.grid import make_grid
from pnp_da_pinn.conservative import solve_reference, physical_current
from pnp_da_pinn.revision_training import train_reference_surrogate, predict

OUT = ROOT / 'outputs/revision'


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def metrics(s, g, c):
    current = physical_current(s, g, c)
    return dict(residual=s.residual, converged=s.converged, current_pA=float(current.mean() * 1e12),
                current_range_pA=float(np.ptp(current) * 1e12),
                conservation=float(np.ptp(current) / max(np.max(np.abs(current)), 1e-30))
                if abs(c.physics.applied_voltage_V) > 1e-12 else np.nan,
                runtime_s=s.runtime_s, wall_cation=float(s.c_plus[g.r >= 0.8].mean()),
                voltage_mV=c.physics.applied_voltage_V * 1000, bulk=c.physics.bulk_concentration_mol_m3,
                wall_charge=c.physics.wall_surface_charge_strength, radius_nm=c.geometry.radius_nm)


def reference(c, tol=1e-9):
    g = make_grid(c.geometry)
    start = time.perf_counter()
    s = solve_reference(c, g, tolerance=tol)
    s.runtime_s = time.perf_counter() - start
    return s, g


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--modes', nargs='+', default=['physics_only', 'data_assimilated', 'wall_adaptive'])
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    c = load_config(ROOT / 'configs/surface_wall_0p036mC.yaml')
    environment = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                       platform=platform.platform(), processor=platform.processor(), torch_threads=2,
                       dtype='float64', device='CPU', steps=args.steps, seeds=args.seeds, config=asdict(c))
    if sys.platform == 'win32':
        environment['cpu_models'] = subprocess.check_output(['powershell', '-NoProfile', '-Command',
                     '(Get-CimInstance Win32_Processor).Name'], text=True).strip()
    (OUT / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    s, g = reference(c)
    np.savez_compressed(OUT / 'reference.npz', r=g.r, z=g.z, psi=s.psi, c_plus=s.c_plus, c_minus=s.c_minus)
    studies = []
    for nr, nz in [(15,45), (21,61), (31,91), (41,121)]:
        cfg = replace(c, geometry=replace(c.geometry, nr=nr, nz=nz))
        ss, gg = reference(cfg)
        np.savez_compressed(OUT / f'grid_{nr}_{nz}.npz', r=gg.r, z=gg.z, psi=ss.psi, c_plus=ss.c_plus, c_minus=ss.c_minus)
        studies.append(dict(nr=nr, nz=nz, **metrics(ss, gg, cfg)))
    fine = np.load(OUT / 'grid_41_121.npz')
    for row in studies:
        dat = np.load(OUT / f"grid_{row['nr']}_{row['nz']}.npz")
        rr, zz = np.meshgrid(dat['r'], dat['z'], indexing='ij')
        points = np.stack((rr, zz), axis=-1)
        for name in ['psi', 'c_plus', 'c_minus']:
            truth = RegularGridInterpolator((fine['r'], fine['z']), fine[name])(points)
            row[name + '_relative_to_finest'] = rel(dat[name], truth)
        truth_charge = RegularGridInterpolator((fine['r'], fine['z']), fine['c_plus'] - fine['c_minus'])(points)
        row['charge_relative_to_finest'] = rel(dat['c_plus'] - dat['c_minus'], truth_charge)
    pd.DataFrame(studies).to_csv(OUT / 'grid_refinement.csv', index=False)
    rows = []
    tight, _ = reference(c, 1e-10)
    for tol in [1e-3, 1e-6, 1e-9, 1e-10]:
        ss, gg = reference(c, tol)
        rows.append(dict(tolerance=tol, **metrics(ss, gg, c), psi_relative_to_tight=rel(ss.psi, tight.psi),
                         charge_relative_to_tight=rel(ss.c_plus-ss.c_minus,tight.c_plus-tight.c_minus)))
    pd.DataFrame(rows).to_csv(OUT / 'tolerance_study.csv', index=False)
    cases = []
    for v in [-.05,-.025,0,.025,.05]:
        cases.append((f'voltage_{v}', replace(c, physics=replace(c.physics, applied_voltage_V=v))))
    for q in [0,-.001,-.005,-.01,-.1]:
        cases.append((f'charge_{q}', replace(c, physics=replace(c.physics, wall_surface_charge_strength=q))))
    for bulk in [10.,100.,1000.]:
        cases.append((f'salt_{bulk}', replace(c, physics=replace(c.physics, bulk_concentration_mol_m3=bulk))))
    for radius in [3.,5.,10.]:
        cases.append((f'radius_{radius}', replace(c, geometry=replace(c.geometry, radius_nm=radius))))
    rows = []
    for name, cfg in cases:
        ss, gg = reference(cfg)
        rows.append(dict(case=name, **metrics(ss,gg,cfg)))
        np.savez_compressed(OUT / f'{name}.npz', r=gg.r, z=gg.z, psi=ss.psi, c_plus=ss.c_plus, c_minus=ss.c_minus)
        print(name, f"residual={ss.residual:.2e}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'parameter_sweep.csv', index=False)
    if args.reference_only:
        return
    rows = []
    for seed in args.seeds:
        for mode in args.modes:
            print(f'Training {mode}, seed {seed}', flush=True)
            trained, indices = train_reference_surrogate(s,g,c,use_data=mode != 'physics_only',seed=seed,
                                                        steps=args.steps,wall_adaptive=mode == 'wall_adaptive')
            fields = predict(trained.model,g,c)
            predicted = PnpSolution(*fields,0,np.nan,0,False)
            holdout = np.ones(g.rr.size,dtype=bool)
            holdout[indices] = False
            heldout = holdout.reshape(g.shape)
            row = dict(seed=seed,mode=mode,training_s=trained.runtime_s,reference_s=s.runtime_s)
            for name, a, b in zip(['psi','c_plus','c_minus','charge'], [*fields, fields[1]-fields[2]],
                                  [s.psi,s.c_plus,s.c_minus,s.c_plus-s.c_minus]):
                row[name+'_rel_l2'] = rel(a,b)
                row[name+'_heldout_rel_l2'] = rel(a[heldout],b[heldout])
                row[name+'_max_abs'] = float(np.max(np.abs(a-b)))
                for region,mask in [('wall',g.rr >= .8),('interior',g.rr < .8)]:
                    row[name+'_'+region+'_rel_l2'] = rel(a[mask],b[mask])
            current = physical_current(predicted,g,c)
            true_current = physical_current(s,g,c)
            row['current_rel_l2'] = rel(current,true_current)
            row['current_max_abs_pA'] = float(np.max(np.abs(current-true_current))*1e12)
            row['current_conservation'] = float(np.ptp(current)/np.max(np.abs(current)))
            for _ in range(10):
                predict(trained.model,g,c)
            times = []
            for _ in range(100):
                start = time.perf_counter()
                predict(trained.model,g,c)
                times.append(time.perf_counter()-start)
            row['inference_median_s'] = float(np.median(times))
            row['inference_p95_s'] = float(np.quantile(times,.95))
            rows.append(row)
            stem = f'{mode}_{seed}'
            np.savez_compressed(OUT / f'{stem}.npz',psi=fields[0],c_plus=fields[1],c_minus=fields[2],training_indices=indices)
            torch.save(trained.model.state_dict(), OUT / f'{stem}.pt')
            pd.DataFrame(trained.history).to_csv(OUT / f'{stem}_history.csv',index=False)
            pd.DataFrame(rows).to_csv(OUT / 'surrogate_runs.csv',index=False)
            print({k:v for k,v in row.items() if k in ['mode','seed','charge_rel_l2','current_rel_l2','training_s']}, flush=True)


if __name__ == '__main__':
    main()
