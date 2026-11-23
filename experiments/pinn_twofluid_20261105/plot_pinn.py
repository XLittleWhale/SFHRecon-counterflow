"""Scientific figures from globally frozen PINN evaluation exports only.

Never reads observations or reference.pt and never selects a solution.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[2]/'outputs/PINNTwoFluid_20261105'
CASES=('vn0_0015m_s','vn0_004m_s')
ARMS=('historical_joint','global_spline','pinn')
LABELS=dict(historical_joint='Historical joint (200)',global_spline='Global spline',pinn='PINN')
COLORS=dict(historical_joint='#737373',global_spline='#0072b2',pinn='#d55e00')
SEEDS=(301030,301031,301032)
BUDGETS=(200,600)
REPRESENTATIVE=301030


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    temporary.replace(path)


def guarded_inputs(out=OUT):
    gate=read(out/'selection_frozen.json')
    expected={(c,a,s) for c in CASES for a in ('global_spline','pinn') for s in SEEDS}
    runs=gate.get('runs',[])
    if len(runs)!=12 or {(r['case'],r['arm'],r['seed']) for r in runs}!=expected:
        raise RuntimeError('All 12 field runs must be globally frozen before plotting')
    for r in runs:
        for budget in ('200','600'):
            for mode in ('primary','terminal'):
                item=r['budgets'][budget][mode]
                if sha(item['path'])!=item['sha256']:raise RuntimeError('Frozen field changed')
    from pinn_common import verify_protocol
    protocol=verify_protocol()
    summary=read(out/'PINN_EVALUATION_SUMMARY.json')
    rows=summary.get('runs',[])
    expected_rows={(c,a,s,b,m) for c in CASES for a in ('global_spline','pinn')
                   for s in SEEDS for b in BUDGETS for m in ('primary','terminal')}
    expected_rows|={(c,'historical_joint',s,200,m) for c in CASES for s in SEEDS for m in ('primary','terminal')}
    actual={(r['case'],r['arm'],r['seed'],int(r['budget']),r['mode']) for r in rows}
    if len(rows)!=60 or actual!=expected_rows:raise RuntimeError('Need all 60 completed evaluation rows')
    manifest=read(out/'FIELD_MANIFEST.json')
    for item in manifest['artifacts']:
        if sha(item['path'])!=item['sha256']:raise RuntimeError('Field export changed: '+item['path'])
    return protocol,summary,manifest


def subset(rows,case,arm,budget,mode='primary'):
    effective=200 if arm=='historical_joint' else budget
    return [r for r in rows if (r['case'],r['arm'],int(r['budget']),r['mode'])==(case,arm,effective,mode)]


def artifact(manifest,case,arm=None,budget=200,mode='primary'):
    for r in manifest['artifacts']:
        if r['case']!=case:continue
        if arm is None and r['kind']=='reference_npz':return Path(r['path'])
        if arm is not None and r['kind']=='field_npz' and r['arm']==arm and int(r['budget'])==(200 if arm=='historical_joint' else budget) and r['mode']==mode:
            return Path(r['path'])
    raise KeyError((case,arm,budget,mode))


def save(fig,out,name,records,description):
    fig.suptitle(description,fontsize=12)
    for extension in ('png','svg'):
        path=out/'figures'/f'{name}.{extension}'
        path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=150,bbox_inches='tight')
        records.append(dict(kind='figure',name=name,format=extension,path=str(path.resolve()),
                            sha256=sha(path),description=description))
    plt.close(fig)


def velocity_maps(rows,manifest,out,records):
    for case in CASES:
        ref=np.load(artifact(manifest,case),allow_pickle=False)
        for budget in BUDGETS:
            fields={arm:np.load(artifact(manifest,case,arm,budget),allow_pickle=False) for arm in ARMS}
            for fluid,fluid_name in enumerate(('normal','superfluid')):
                frames=(0,50,95)
                speeds={'Reference':np.linalg.norm(ref['velocity_m_s'][:,fluid],axis=-1)*1000}
                speeds.update({LABELS[arm]:np.linalg.norm(data['velocity_m_s'][:,fluid],axis=-1)*1000 for arm,data in fields.items()})
                extent=(float(ref['x_m'].min()*1000),float(ref['x_m'].max()*1000),
                        float(ref['y_m'].min()*1000),float(ref['y_m'].max()*1000))
                lo=min(float(value[list(frames)].min()) for value in speeds.values())
                hi=max(float(value[list(frames)].max()) for value in speeds.values())
                fig,axes=plt.subplots(4,3,figsize=(11,12),layout='constrained')
                for i,(name,values) in enumerate(speeds.items()):
                    for j,frame in enumerate(frames):
                        ax=axes[i,j];im=ax.imshow(values[frame],origin='lower',extent=extent,vmin=lo,vmax=hi,cmap='viridis',aspect='equal')
                        ax.set_title(f'{name}; relative t={frame} ms',fontsize=9)
                        ax.set_xlabel('x [mm]');ax.set_ylabel('y [mm]')
                fig.colorbar(im,ax=axes,shrink=.7,label='Speed [mm/s]')
                save(fig,out,f'{case}_budget{budget}_{fluid_name}_primary_speed',records,
                     f'{case}; {fluid_name}; {budget} updates; V-selected; seed {REPRESENTATIVE}')
                errors={LABELS[arm]:np.linalg.norm(data['velocity_error_m_s'][:,fluid],axis=-1)*1000 for arm,data in fields.items()}
                hi=max(float(value[list(frames)].max()) for value in errors.values())
                fig,axes=plt.subplots(3,3,figsize=(11,9),layout='constrained')
                for i,(name,values) in enumerate(errors.items()):
                    for j,frame in enumerate(frames):
                        ax=axes[i,j];im=ax.imshow(values[frame],origin='lower',extent=extent,vmin=0,vmax=max(hi,1e-12),cmap='magma',aspect='equal')
                        ax.set_title(f'{name}; relative t={frame} ms',fontsize=9)
                        ax.set_xlabel('x [mm]');ax.set_ylabel('y [mm]')
                fig.colorbar(im,ax=axes,shrink=.75,label='Velocity error magnitude [mm/s]')
                save(fig,out,f'{case}_budget{budget}_{fluid_name}_primary_error',records,
                     f'{case}; {fluid_name} error; {budget} updates; V-selected; seed {REPRESENTATIVE}')
            for value in fields.values():value.close()
        ref.close()


def time_curves(rows,out,records):
    for case in CASES:
        for budget in BUDGETS:
            fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
            for fluid,name in enumerate(('Normal','Superfluid')):
                ax=axes[fluid]
                for arm in ARMS:
                    for mode,style in (('primary','-'),('terminal','--')):
                        values=np.array([r['field']['time_rms_mm_s'] for r in subset(rows,case,arm,budget,mode)])[:,:,fluid]
                        mean=values.mean(0)
                        ax.plot(np.arange(96),mean,style,color=COLORS[arm],label=f'{LABELS[arm]} / {mode}')
                        if mode=='primary':ax.fill_between(np.arange(96),values.min(0),values.max(0),color=COLORS[arm],alpha=.12)
                ax.set(xlabel='Time from window start [ms]',ylabel='Velocity RMS error [mm/s]',title=name)
                ax.grid(alpha=.2)
            axes[1].legend(fontsize=8)
            save(fig,out,f'{case}_budget{budget}_time_rms',records,
                 f'{case}; {budget} updates; 3-seed mean, primary range; historical always 200')


def metric(row,index):
    if index<2:return row['field']['initial_rms_mm_s'][index]
    if index<4:return row['field']['rms_mm_s'][index-2]
    return row['observations']['H']['free' if index==4 else 'cap']


def overviews(rows,out,records):
    titles=('Initial normal RMS','Initial superfluid RMS','Window normal RMS','Window superfluid RMS','H free NLL','H captured NLL')
    for budget in BUDGETS:
        fig,axes=plt.subplots(2,6,figsize=(18,7),layout='constrained')
        for i,case in enumerate(CASES):
            for k,title in enumerate(titles):
                ax=axes[i,k]
                for j,arm in enumerate(ARMS):
                    for mode,offset,marker in (('primary',-.11,'o'),('terminal',.11,'x')):
                        vals=np.array([metric(r,k) for r in subset(rows,case,arm,budget,mode)])
                        ax.scatter(j+offset+np.linspace(-.025,.025,len(vals)),vals,color=COLORS[arm],marker=marker,s=25,alpha=.65)
                        ax.scatter(j+offset,vals.mean(),color='black',marker='_',s=130,zorder=5)
                ax.set_title(title,fontsize=9);ax.grid(axis='y',alpha=.2)
                ax.set_xticks(range(3),['Historical\n200','Spline','PINN'],fontsize=8)
                ax.set_ylabel((case+'\n' if k==0 else '')+('mm/s' if k<4 else 'NLL'))
        save(fig,out,f'overview_budget{budget}',records,
             f'{budget}-update comparison; circles V-selected, crosses terminal, black marks seed means; lower is better')


def diagnostics(rows,summary,out,records):
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for i,kind in enumerate(('free','cap')):
        ax=axes[i]
        for j,arm in enumerate(ARMS):
            values=[r['heldout_degree28'][kind]-r['observations']['H'][kind] for r in rows if r['arm']==arm]
            ax.scatter(np.full(len(values),j)+np.linspace(-.1,.1,len(values)),values,color=COLORS[arm],s=20)
        ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(range(3),[LABELS[a] for a in ARMS],fontsize=8)
        ax.set(ylabel='NLL(GH28) - NLL(GH20)',title=kind+'; all selected/terminal/budget rows');ax.grid(alpha=.2)
    save(fig,out,'quadrature_sensitivity',records,'Quadrature sensitivity is descriptive; no post-hoc acceptance threshold')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    used=set();costs={arm:[] for arm in ARMS}
    for r in rows:
        key=(r['case'],r['arm'],r['seed'])
        if r['mode']!='terminal' or int(r['budget'])!=(200 if r['arm']=='historical_joint' else 600) or key in used:continue
        used.add(key)
        wall=r.get('wall_seconds',r.get('cost',{}).get('wall_seconds'))
        cpu=r.get('cpu_seconds',r.get('cost',{}).get('cpu_seconds'))
        if wall is None or cpu is None:raise RuntimeError('Missing exact terminal run cost')
        costs[r['arm']].append((float(wall),float(cpu)))
    for i,key in enumerate(('Wall sum [min]','CPU sum [min]')):
        ax=axes[i]
        values=[sum(v[i] for v in costs[a])/60 for a in ARMS]
        ax.bar(range(3),values,color=[COLORS[a] for a in ARMS])
        ax.set_xticks(range(3),['Historical 200\nprevious cost','Spline 600\nnew cost','PINN 600\nnew cost']);ax.set_ylabel(key)
        for j,value in enumerate(values):ax.text(j,value,f'{value:.1f}',ha='center',va='bottom',fontsize=9)
    save(fig,out,'compute_cost',records,'Unique terminal-run costs; 200-step checkpoints are nested in 600; worker wall sum is not elapsed pipeline time')


def main():
    protocol,summary,manifest=guarded_inputs()
    rows=summary['runs'];records=[]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    velocity_maps(rows,manifest,OUT,records)
    time_curves(rows,OUT,records);overviews(rows,OUT,records);diagnostics(rows,summary,OUT,records)
    if len(records)!=48:raise RuntimeError('Expected 24 figure groups, each PNG and SVG')
    dump(OUT/'PLOT_MANIFEST.json',dict(complete=True,protocol_fingerprint=protocol['fingerprint'],
         script_sha256=sha(__file__),evaluation_summary_sha256=sha(OUT/'PINN_EVALUATION_SUMMARY.json'),
         field_manifest_sha256=sha(OUT/'FIELD_MANIFEST.json'),selection_gate_sha256=sha(OUT/'selection_frozen.json'),
         representative_seed=REPRESENTATIVE,figure_groups=24,artifacts=records,
         target_reference_read=False,observations_read=False,new_model_selection=False))
    print('PLOTS_COMPLETE',24,flush=True)


if __name__=='__main__':main()
