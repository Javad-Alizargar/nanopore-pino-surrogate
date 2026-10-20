"""Preserve the submitted workflow artwork; plot only corrected numerical data."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'outputs/revision'
OUT = DATA/'figures'
PANELS_ONLY='--panels-only' in sys.argv
sys.path.insert(0,str(ROOT/'src'))
from pnp_da_pinn.config import load_config
from pnp_da_pinn.grid import make_grid
from pnp_da_pinn.baseline import PnpSolution
from pnp_da_pinn.conservative import physical_current


def save(fig,name):
    if not PANELS_ONLY:
        fig.savefig(OUT/(name+'.png'),dpi=300,facecolor='white')
        fig.savefig(OUT/(name+'.pdf'),facecolor='white')
    save_panels(fig,name)
    plt.close(fig)


def save_panels(fig,name):
    counts={'Figure_2_reference_fields':4,'Figure_3_grid_refinement':3,
            'Figure_4_surrogate_errors':6,'Figure_5_surface_response':4,
            'Figure_6_ablation':3,'Figure_S1_extended_sweeps':3}
    folder=OUT/'panels';folder.mkdir(exist_ok=True)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    if name=='Figure_1_workflow':
        bounds=[(0,0,507,350),(814,0,480,356),(1530,0,518,349),
                (924,376,590,319),(0,353,532,342)]
        width,height=fig.get_size_inches()
        for letter,(x,y,w,h) in zip('ABCDE',bounds):
            box=Bbox.from_bounds(x/2048*width,(695-y-h)/695*height,w/2048*width,h/695*height)
            fig.savefig(folder/f'Figure_1{letter}.pdf',bbox_inches=box,facecolor='white')
        return
    count=counts[name]
    for letter,ax in zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',fig.axes[:count]):
        boxes=[ax.get_tightbbox(renderer)]
        for other in fig.axes[count:]:
            colorbar=getattr(other,'_colorbar',None)
            if colorbar is not None and colorbar.mappable.axes is ax:
                boxes.append(other.get_tightbbox(renderer))
        box=Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(.08)
        prefix='_'.join(name.split('_')[:2])
        fig.savefig(folder/f'{prefix}{letter}.pdf',bbox_inches=box,facecolor='white')


def workflow(ref,pred):
    # Coordinates refer to the submitted artwork at a 2048 x 695 reference size.
    # The background remains unchanged outside explicitly replaced scientific insets.
    fig = plt.figure(figsize=(20,20*695/2048))
    canvas = fig.add_axes([0,0,1,1])
    canvas.imshow(plt.imread(ROOT/'assets/workflow_original.png'),extent=(0,2048,695,0),aspect='auto')
    canvas.set(xlim=(0,2048),ylim=(695,0))
    canvas.axis('off')

    def erase(x,y,w,h):
        canvas.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor='none'))

    def label(x,y,text,size=12,**kwargs):
        canvas.text(x,y,text,fontsize=size,ha=kwargs.pop('ha','center'),va='center',**kwargs)

    def inset(x,y,w,h):
        return fig.add_axes([x/2048,1-(y+h)/695,w/2048,h/695])

    def thumbnail(x,y,w,h,values):
        ax=inset(x,y,w,h)
        ax.pcolormesh(ref['z'],ref['r'],values,shading='auto',cmap='viridis')
        ax.set_xticks([]);ax.set_yticks([])
        return ax

    # Panel B preserves its discretization schematic, with corrected field maps.
    erase(1165,82,108,181)
    for y,key in [(91,'psi'),(148,'c_plus'),(205,'c_minus')]:
        thumbnail(1174,y,87,44,ref[key])
    erase(857,273,400,83)
    label(1057,294,'Equilibrium tests  |  Conservation',size=11,color='#235d32')
    label(1057,317,'Grid and tolerance checks',size=11,color='#174a82')
    label(1057,343,'Verified numerical reference',size=12,style='italic')

    # Panel C shows the exact saved training-node subset, not illustrative dots.
    thumbnail(1558,130,116,83,ref['c_plus']-1)
    ax=thumbnail(1746,106,253,143,ref['c_plus']-1)
    rr,zz=np.meshgrid(ref['r'],ref['z'],indexing='ij')
    indices=pred['training_indices']
    ax.scatter(zz.ravel()[indices],rr.ravel()[indices],s=4,c='white',edgecolors='#175329',linewidths=.35)
    erase(1550,279,492,68)
    label(1798,291,'500 distinct anchors / 1281 nodes (39%)',size=11)
    label(1798,324,'Saved reference fields and sampling locations',size=10,style='italic')
    erase(1578,233,86,31)
    label(1621,250,r'$c_+-1$',size=11)

    # Panel D keeps the network artwork; only the scientific labels change.
    erase(930,378,495,38)
    label(1176,397,'D. Exact-reservoir DA-PINN',size=17,weight='bold')
    erase(1015,632,360,63)
    label(1195,643,'Four hidden layers, 64 SiLU units each',size=10)
    label(1195,669,'PNP + wall/axis + field/charge losses',size=11,
          bbox={'boxstyle':'round,pad=.35','facecolor':'white','edgecolor':'#444444'})
    erase(1363,557,68,36)
    label(1442,610,'Exact reservoir\nDirichlet values',size=8,color='#173c89')

    # Panel E replaces every old result, including the unrelated hourglass maps.
    erase(0,353,507,342)
    label(253,378,'E. Fixed-state field prediction',size=16,weight='bold')
    for x,key,title in [(28,'psi',r'$\psi$'),(153,'c_plus',r'$c_+$'),(278,'c_minus',r'$c_-$')]:
        ax=thumbnail(x,430,85,130,pred[key])
        ax.set_title(title,fontsize=11,pad=5)
        ax.set_xticks([0,1]);ax.set_xlabel('z/L',fontsize=8,labelpad=1)
        ax.set_yticks([0,1]);ax.tick_params(labelsize=7,length=2,pad=1)
        if x==28:ax.set_ylabel('r/R',fontsize=8,labelpad=1)
        bar=fig.colorbar(ax.collections[0],cax=inset(x+88,430,5,130))
        bar.ax.tick_params(labelsize=6,length=2,pad=1)
        bar.ax.yaxis.get_offset_text().set_fontsize(6)
    config=load_config(ROOT/'configs/surface_wall_0p036mC.yaml')
    grid=make_grid(config.geometry)
    solution=PnpSolution(pred['psi'],pred['c_plus'],pred['c_minus'],0,0.,0.,True)
    current=physical_current(solution,grid,config)*1e12
    ax=inset(406,430,89,130)
    ax.plot((ref['z'][:-1]+ref['z'][1:])/2,current,color='#22569c',lw=1.8)
    ax.set(xlabel='z/L',title='Current (pA)',xlim=(0,1))
    ax.set_xticks([0,1]);ax.tick_params(labelsize=7,length=2,pad=1)
    ax.xaxis.label.set_size(8);ax.title.set_size(10)
    ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    ax.yaxis.tick_right()
    timing=json.loads((DATA/'timing_summary.json').read_text())
    ms=timing['pinn_full_field']['median_s']*1000
    label(253,611,f'{ms:.3f} ms median full-field evaluation',size=12,color='#235d32',
          bbox={'boxstyle':'round,pad=.4','facecolor':'#edf6e9','edgecolor':'#407437'})
    label(253,645,'Timing excludes current calculation and training',size=10)
    label(253,672,'Fixed operating state; no generalization claim',size=10,style='italic')
    save(fig,'Figure_1_workflow')


def surface_response(ref,sweep):
    with plt.rc_context({'font.size':12,'axes.spines.top':True,'axes.spines.right':True}):
        fig,axes=plt.subplots(2,2,figsize=(11.5,8),layout='constrained')
        iv=sweep[sweep['case'].str.startswith('voltage')].sort_values('voltage_mV')
        ax=axes[0,0]
        ax.plot(iv.voltage_mV,iv.current_pA,'o-',color='black',lw=2)
        ax.axhline(0,color='.7',lw=1);ax.axvline(0,color='.7',lw=1)
        ax.set(title='Current-voltage response',xlabel='Applied voltage (mV)',ylabel='Mean current (pA)')
        charge=sweep[sweep['case'].isin(['charge_0','charge_-0.001','charge_-0.005','charge_-0.01'])].sort_values('wall_charge',ascending=False)
        neutral=charge.loc[charge.wall_charge==0,'wall_cation'].iloc[0]
        axes[0,1].plot(-charge.wall_charge,charge.wall_cation-neutral,'o-',color='#507fa9',lw=2)
        axes[0,1].set(title='Surface-charge ion redistribution',xlabel=r'$|q_w|$ (dimensionless)',ylabel=r'Near-wall cation enrichment, $\Delta\langle c_+\rangle$')
        axes[0,1].ticklabel_format(axis='y',style='plain',useOffset=False)
        charged=np.load(DATA/'charge_-0.01.npz')
        uncharged=np.load(DATA/'charge_0.npz')
        difference=charged['c_plus']-uncharged['c_plus']
        limit=float(np.abs(difference).max())
        im=axes[1,0].contourf(ref['z'],ref['r'],difference,levels=np.linspace(-limit,limit,31),cmap='coolwarm')
        fig.colorbar(im,ax=axes[1,0],label=r'$\Delta c_+$',format='%.4f')
        axes[1,0].set(title='Cation redistribution by wall charge',xlabel='z/L',ylabel='r/R')
        quality=charge.sort_values('wall_charge')
        x=np.arange(len(quality))
        axes[1,1].bar(x-.18,quality.residual,.36,color='#59a452',label='Fixed-point residual')
        axes[1,1].bar(x+.18,quality.conservation,.36,color='#e6565c',label='Current nonuniformity')
        axes[1,1].axhline(1e-9,color='#555555',ls='--',lw=1,label='Acceptance threshold')
        axes[1,1].set(yscale='log',title='Solver quality across charge sweep',ylabel='Dimensionless diagnostic',xlabel=r'$|q_w|$ (dimensionless)',ylim=(1e-14,4e-9))
        axes[1,1].set_xticks(x,[f'{abs(v):g}' for v in quality.wall_charge])
        axes[1,1].legend(fontsize=8,loc='upper right')
        for ax,letter in zip(axes.ravel(),'ABCD'):
            ax.text(.025,.94,letter,transform=ax.transAxes,weight='bold',va='top',fontsize=15,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':2})
        save(fig,'Figure_5_surface_response')
        fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
        all_charge=sweep[sweep['case'].str.startswith('charge')].sort_values('wall_charge',ascending=False)
        axes[0].plot(-all_charge.wall_charge,all_charge.wall_cation-neutral,'o-',color='#507fa9')
        axes[0].set(title='A  Extended charge sweep',xlabel=r'$|q_w|$',ylabel=r'$\Delta\langle c_+\rangle$')
        salt=sweep[sweep['case'].str.startswith('salt')].sort_values('bulk')
        axes[1].semilogx(salt.bulk,salt.wall_cation-1,'o-',color='#a65143')
        axes[1].set(title='B  Salt sweep',xlabel=r'Bulk concentration (mol m$^{-3}$)',ylabel=r'$\langle c_+\rangle-1$')
        radius=sweep[sweep['case'].str.startswith('radius')].sort_values('radius_nm')
        axes[2].plot(radius.radius_nm,radius.current_pA,'o-',color='#6f5b92')
        axes[2].set(title='C  Radius sweep',xlabel='Radius (nm)',ylabel='Current (pA)')
        save(fig,'Figure_S1_extended_sweeps')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    ref=np.load(DATA/'reference.npz')
    workflow(ref,np.load(DATA/'data_assimilated_42.npz'))
    surface_response(ref,pd.read_csv(DATA/'parameter_sweep.csv'))
