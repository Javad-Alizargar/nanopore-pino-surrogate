"""Independent diagnostics and descriptive seed summaries for the revision."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import RegularGridInterpolator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pnp_da_pinn.config import load_config
from pnp_da_pinn.grid import make_grid
from pnp_da_pinn.conservative import ConservativePNP, physical_current, solve_reference
from pnp_da_pinn.baseline import PnpSolution, current_scale_A
from pnp_da_pinn.sg_solver import annular_control_volume_weights, bernoulli, sg_face_flux_z
from pnp_da_pinn.scales import voltage_to_psi
from pnp_da_pinn.pinn import PnpPINN, AnsatzConfig, predict_fields, grad, pnp_residuals
from pnp_da_pinn.revision_training import predict

OUT=ROOT/'outputs/revision'


def main():
    torch.set_num_threads(2)
    c=load_config(ROOT/'configs/surface_wall_0p036mC.yaml')
    g=make_grid(c.geometry)
    ref=np.load(OUT/'reference.npz')
    solution=PnpSolution(ref['psi'],ref['c_plus'],ref['c_minus'],0,0,0,True)
    p=ConservativePNP(g,c.scales.poisson_coupling,voltage_to_psi(c.physics.applied_voltage_V,c.scales),c.physics.wall_surface_charge_strength)
    rhs=p.load.copy()
    rhs[:,1:-1]+=c.scales.poisson_coupling*(solution.c_plus-solution.c_minus)[:,1:-1]
    raw=p.poisson_matrix @ solution.psi.ravel()-rhs.ravel()
    details={'poisson_raw_max':float(np.max(np.abs(raw[p.free]))),
             'poisson_diagonal_scaled_max':float(np.max(np.abs(raw[p.free]/p.poisson_matrix.diagonal()[p.free])))}
    for sign,name,field in [(1,'cation',solution.c_plus),(-1,'anion',solution.c_minus)]:
        delta=sign*(solution.psi.ravel()[p.col]-solution.psi.ravel()[p.row])
        matrix=p.matrix(bernoulli(delta),bernoulli(-delta))
        rhs_np=np.zeros(p.n)
        rhs_np[p.boundary]=1
        raw_np=matrix @ field.ravel()-rhs_np
        details[name+'_diagonal_scaled_max']=float(np.max(np.abs(raw_np[p.free]/matrix.diagonal()[p.free])))
        flux=np.sum(sg_face_flux_z(field,solution.psi,g,sign)*annular_control_volume_weights(g)[:,None],axis=0)
        details[name+'_relative_flux_range']=float(np.ptp(flux)/np.max(np.abs(flux)))
    (OUT/'independent_reference_diagnostics.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    sweep=pd.read_csv(OUT/'parameter_sweep.csv')
    sweep['surface_charge_mC_m2']=sweep.wall_charge*c.scales.permittivity_F_m*c.scales.thermal_voltage_V/(sweep.radius_nm*1e-9)*1000
    sweep.to_csv(OUT/'parameter_sweep.csv',index=False)
    runs=pd.read_csv(OUT/'surrogate_runs.csv')
    assert len(runs)==9 and not runs.duplicated(['seed','mode']).any()
    rows=[]
    wr=annular_control_volume_weights(g)[:,None]
    wz=np.ones((1,g.z.size))*g.dz
    wz[:,[0,-1]]*=.5
    weights=wr*wz
    rho=ref['c_plus']-ref['c_minus']
    rng=np.random.default_rng(20261013)
    rv=torch.tensor(rng.uniform(.001,.999,(1000,1)),dtype=torch.float64)
    zv=torch.tensor(rng.uniform(.001,.999,(1000,1)),dtype=torch.float64)
    z_boundary=(1-np.cos(np.pi*(np.arange(512)+.5)/512))/2
    for _,run in runs.iterrows():
        name=f'{run["mode"]}_{int(run.seed)}'
        fields=np.load(OUT/f'{name}.npz')
        model=PnpPINN().double()
        model.load_state_dict(torch.load(OUT/f'{name}.pt',map_location='cpu',weights_only=True))
        ansatz=AnsatzConfig(voltage_to_psi(c.physics.applied_voltage_V,c.scales))
        row={'mode':run['mode'],'seed':int(run.seed)}
        heldout=np.ones(g.rr.size,dtype=bool)
        heldout[fields['training_indices']]=False
        heldout=heldout.reshape(g.shape)&(g.zz>0)&(g.zz<1)
        row['training_nodes']=len(fields['training_indices'])
        row['heldout_interior_nodes']=int(heldout.sum())
        for key in ['psi','c_plus','c_minus','charge']:
            a=fields[key] if key!='charge' else fields['c_plus']-fields['c_minus']
            b=ref[key] if key!='charge' else rho
            row[key+'_volume_rel_l2']=float(np.sqrt(np.sum(weights*(a-b)**2)/np.sum(weights*b*b)))
            row[key+'_heldout_interior_rel_l2']=float(np.linalg.norm((a-b)[heldout])/np.linalg.norm(b[heldout]))
        diff=fields['c_plus']-fields['c_minus']-rho
        for label,mask in [('wall',g.rr>=.8),('interior',g.rr<.8),('corner',(g.rr>=.8)&((g.zz<.05)|(g.zz>.95)))]:
            row['charge_'+label+'_rms_abs']=float(np.sqrt(np.mean(diff[mask]**2)))
            row['charge_'+label+'_ref_rms']=float(np.sqrt(np.mean(rho[mask]**2)))
        res=pnp_residuals(model,rv,zv,ansatz,c.scales.poisson_coupling,g.radial_metric)
        for key in ['poisson','np_plus','np_minus']:
            row[key+'_test_rms']=float(res[key].detach().square().mean().sqrt())
        for radius,label in [(0,'axis'),(1,'wall')]:
            rb=torch.full((512,1),float(radius),dtype=torch.float64,requires_grad=True)
            zb=torch.tensor(z_boundary[:,None],dtype=torch.float64)
            psi,cp,cm=predict_fields(model,rb,zb,ansatz)
            dp=grad(psi,rb)
            bc=[dp-(c.physics.wall_surface_charge_strength if radius else 0),grad(cp,rb)+cp*dp,grad(cm,rb)-cm*dp]
            for name_bc,values in zip(['potential_derivative','cation_flux','anion_flux'],bc):
                row[label+'_'+name_bc+'_rms']=float(values.detach().square().mean().sqrt())
                row[label+'_'+name_bc+'_max']=float(values.detach().abs().max())
        nodes,quad=np.polynomial.legendre.leggauss(80)
        radius=(nodes+1)/2
        rr,zz=np.meshgrid(radius,.5*(g.z[:-1]+g.z[1:]),indexing='ij')
        rz=torch.tensor(rr.reshape(-1,1),dtype=torch.float64)
        zz_t=torch.tensor(zz.reshape(-1,1),dtype=torch.float64,requires_grad=True)
        psi,cp,cm=predict_fields(model,rz,zz_t,ansatz)
        dp=grad(psi,zz_t)
        jp=-(grad(cp,zz_t)+cp*dp).detach().numpy().reshape(rr.shape)
        jm=-(grad(cm,zz_t)-cm*dp).detach().numpy().reshape(rr.shape)
        dp_const,dm_const=c.physics.diffusion_cation_m2_s,c.physics.diffusion_anion_m2_s
        current=2*np.pi*np.sum((dp_const*jp-dm_const*jm)*(quad*radius/2)[:,None],axis=0)*current_scale_A(c)/((dp_const+dm_const)/2)
        truth=physical_current(solution,g,c)
        row['autograd_current_rel_l2']=float(np.linalg.norm(current-truth)/np.linalg.norm(truth))
        row['autograd_current_relative_range']=float(np.ptp(current)/np.max(np.abs(current)))
        raw=model(rv,zv).detach().numpy()
        exponent=zv.numpy()*(1-zv.numpy())*raw[:,1:]
        row['log_concentration_max_abs']=float(np.max(np.abs(exponent)))
        row['clipping_detected']=bool(np.any(np.abs(exponent)>=20))
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'independent_surrogate_diagnostics.csv',index=False)
    summary=runs.groupby('mode').agg({key:['mean','std','min','max'] for key in
             ['psi_rel_l2','c_plus_rel_l2','c_minus_rel_l2','charge_rel_l2','current_rel_l2','training_s','inference_median_s']})
    summary.to_csv(OUT/'seed_summary.csv')
    differences=[]
    for key in ['psi_rel_l2','c_plus_rel_l2','c_minus_rel_l2','charge_rel_l2','current_rel_l2']:
        wide=runs.pivot(index='seed',columns='mode',values=key)
        for mode in ['data_assimilated','wall_adaptive']:
            delta=wide[mode]-wide['physics_only']
            differences.append(dict(quantity=key,comparison=mode+' minus physics_only',n=3,
                                    mean_difference=delta.mean(),sample_sd=delta.std(ddof=1),
                                    min_difference=delta.min(),max_difference=delta.max()))
    pd.DataFrame(differences).to_csv(OUT/'paired_differences.csv',index=False)
    # Repeat evaluation benchmarks after training has finished, on the same CPU.
    model=PnpPINN().double()
    model.load_state_dict(torch.load(OUT/'data_assimilated_42.pt',weights_only=True))
    interpolators=[RegularGridInterpolator((g.r,g.z),ref[key]) for key in ['psi','c_plus','c_minus']]
    points=np.stack((g.rr,g.zz),axis=-1)
    timing={}
    for label,fn in [('pinn_full_field',lambda:predict(model,g,c)),
                     ('cached_reference_interpolation',lambda:[f(points) for f in interpolators])]:
        for _ in range(10):fn()
        samples=[]
        for _ in range(100):
            start=time.perf_counter(); fn(); samples.append(time.perf_counter()-start)
        timing[label]={'median_s':float(np.median(samples)),'p95_s':float(np.quantile(samples,.95)),'n':100}
    samples=[]
    for _ in range(5):
        start=time.perf_counter(); solve_reference(c,g); samples.append(time.perf_counter()-start)
    timing['fresh_reference_solve']={'median_s':float(np.median(samples)),'min_s':min(samples),'max_s':max(samples),'n':5}
    (OUT/'timing_summary.json').write_text(json.dumps(timing,indent=2),encoding='utf-8')
    print(json.dumps(timing,indent=2))


if __name__=='__main__':main()
