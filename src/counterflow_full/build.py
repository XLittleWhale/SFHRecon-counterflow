"""Local I/O-only builder; no network training or fluid simulation is run here."""
import argparse
import json
import shutil
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from counterflow.io import dump, save, load, sha
from .core import config


def records(x, state, cfg):
    start,end=cfg['start'],cfg['end'];rows={k:[] for k in ('frame','pid','state','x','w','h','target')}
    assert x.ndim==3 and len(x)>end and start>=5
    audit=dict(total_intervals=(end-start)*x.shape[1],state_transition_intervals=0,nonfinite_intervals=0)
    for t in range(start,end):
        w=(x[t]-x[t-1])/.001
        # Half-ms history query uses only real positions <=t, linear predictor thereafter.
        tt=t+np.arange(-5,1)+.5;hp=[]
        for v in tt:
            if v>t:hp.append(x[t]+(v-t)*.001*w)
            else:
                k=int(np.floor(v));hp.append(x[k]+(v-k)*(x[k+1]-x[k]))
        hp=np.asarray(hp)
        h=np.concatenate((np.diff(hp,axis=0)/.001,(hp[1:]-hp[-1])/.005),-1).transpose(1,0,2)
        finite=np.isfinite(h).all((1,2))&np.isfinite(x[t+1]).all(1)
        stable=state[t]==state[t+1]
        audit['state_transition_intervals']+=int((~stable).sum())
        audit['nonfinite_intervals']+=int((~finite).sum())
        ids=np.where(finite&stable&np.isin(state[t],[0,1]))[0]
        values=dict(frame=np.full(len(ids),t-start),pid=ids,state=state[t,ids],
                    x=x[t,ids]+.0005*w[ids],w=w[ids],h=h[ids],target=x[t+1,ids]-x[t,ids])
        for k,v in values.items():rows[k].append(v)
    z={k:np.concatenate(v) for k,v in rows.items()}
    audit['valid_intervals']=len(z['pid']);audit['unique_pids']=len(np.unique(z['pid']))
    audit['counts']={f'{kind}_{group}':dict(intervals=int(m.sum()),pids=int(len(np.unique(z['pid'][m]))))
        for kind,s in [('free',0),('cap',1)] for group in ('train','validation','test')
        for m in [(z['state']==s)&((z['pid']%5>=2) if group=='train' else z['pid']%5==(1 if group=='validation' else 0))]}
    return z,audit


def build(source, dest):
    import counterflow
    cfg=config();dest.mkdir(parents=True,exist_ok=True)
    # Build a small standalone distribution, not a dependency on the old outputs tree.
    for folder,files in [('counterflow',['__init__.py','io.py','observation.py','dynamics.py','vtk.py']),
                         ('counterflow_full',[p.name for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.json','.md')])]:
        out=dest/folder;out.mkdir(exist_ok=True)
        src=Path(counterflow.__file__).parent if folder=='counterflow' else Path(__file__).parent
        for name in files:shutil.copy2(src/name,out/name)
    shutil.copytree(Path(__file__).parent/'resources',dest/'counterflow_full'/'resources',dirs_exist_ok=True)
    for name in ('README.md','run_all.py','requirements.txt'):
        shutil.copy2(Path(__file__).parent/'assets'/name,dest/name)
    resources=Path(__file__).parent/'resources'
    template=json.loads((resources/'physics_4mm.json').read_text(encoding='utf8'))
    shutil.copy2(resources/'operator.pt',dest/'operator.pt')
    reports=[]
    for name in cfg['train']+[cfg['validation'],cfg['test']]:
        p=source/name;target=dest/'data'/name;target.mkdir(parents=True,exist_ok=True)
        meta=json.loads((p/'metadata.json').read_text(encoding='utf8'))
        physics=meta['config'];ten=meta['tension_closure']
        for key in ('temperature_K','particle_radius_m','particle_density_kg_m3'):
            if not np.isclose(physics[key],template['physics'][key]):raise ValueError('Physical constants mismatch: '+key)
        c=dict(physics=physics,constants=template['constants'],hr=ten['anchor_separation_m']/physics['particle_radius_m'],
               tension_twice_N=2*ten['effective_line_tension_N'])
        dump(target/'physics.json',c)
        with np.load(p/'particle_trajectories.npz') as z:
            x=z['xy_unwrapped_m'].astype(float);times=z['time_s'];pid=z['particle_id']
        if not np.array_equal(pid,np.arange(len(pid))):raise ValueError('Non-contiguous particle IDs require an explicit mapping')
        if not np.allclose(times,np.arange(len(times))*.001,atol=1e-12):raise ValueError('Not native 1ms saved data')
        with np.load(p/'particle_hidden_truth.npz') as z:
            states=z['state'];attached=z['attached_vortex'];anchors=z['anchor_xy_unwrapped_m']
            forces=z['force_tension_xy_N']
            if not np.array_equal(z['time_s'],times):raise ValueError('Hidden/trajectory time mismatch')
        obs,audit=records(x,states,cfg);U=physics['vn_mean_m_s']
        obs['w']/=U;obs['h']/=U;obs['target']/=(U*.001)
        save(target/'observations.pt',{k:torch.from_numpy(v) for k,v in obs.items()})
        frames=obs['frame']+cfg['start'];ids=obs['pid'];cap=obs['state']==1
        delta=anchors-x[:,:,None];delta-=physics['box_m']*np.round(delta/physics['box_m'])
        valid=states==1
        symmetry=float(np.max(np.abs(delta[:,:,0][valid]-delta[:,:,1][valid])))
        if symmetry>1e-10:raise ValueError('One-anchor network invalid for asymmetric anchors')
        geom=ten['effective_line_tension_N']*(delta/np.sqrt(ten['anchor_separation_m']**2+(delta**2).sum(-1,keepdims=True))).sum(2)
        force_error=float(np.max(np.linalg.norm((geom-forces)[valid],axis=-1)))
        # Saved last_tension is the final microscopic midpoint force, whereas
        # anchors are endpoint positions (forward/tethered.py). Audit the gap;
        # it is not an exact identity and is not used to relabel the geometry.
        if force_error>.05*c['tension_twice_N']:raise ValueError('Unexpectedly large endpoint/midpoint tension gap; inspect source')
        q=np.zeros(len(ids),dtype=np.int64);idx=attached[frames,ids]
        for t in range(cfg['start'],cfg['end']):
            ix=np.where(cap&(frames==t))[0]
            with h5py.File(p/f'vortices_{t:04d}.vtkhdf','r') as f:
                signs=f['VTKHDF/PointData/sign'][:]
                # attached_vortex is a current array index in the source solver.
                if (idx[ix]<0).any() or (idx[ix]>=len(signs)).any():raise ValueError('Invalid attached vortex index')
                q[ix]=(signs[idx[ix]]>0).astype(np.int64)
        label=dict(q=torch.from_numpy(q),y=torch.from_numpy(delta[frames,ids,0]/physics['particle_radius_m']))
        # Test latent truth is audit-only: training/inversion never load this file.
        save(target/'latent_truth.pt',label)
        signature=dict(config=cfg['version'],start=cfg['start'],end=cfg['end'],
            files=[[f.name,f.stat().st_size,f.stat().st_mtime_ns] for f in
                   [p/f'field_{t:04d}.vtkhdf' for t in range(cfg['start'],cfg['end']+1)]])
        raw=target/'fields.h5';sig=target/'fields_signature.json'
        reusable=raw.exists() and sig.exists() and json.loads(sig.read_text(encoding='utf8'))==signature
        if not reusable:
            if shutil.disk_usage(dest).free<4*1024**3:raise RuntimeError('Insufficient disk headroom')
            tmp=target/'fields.h5.tmp';started=time.perf_counter()
            with h5py.File(tmp,'w') as out:
                ds=None
                for j,t in enumerate(range(cfg['start'],cfg['end']+1)):
                    with h5py.File(p/f'field_{t:04d}.vtkhdf','r') as f:
                        v=np.stack([f['VTKHDF/PointData/'+key][:-1,:-1,:2] for key in ('vn','vs')],2)
                    if ds is None:
                        ds=out.create_dataset('v',shape=(cfg['end']-cfg['start']+1,)+v.shape,
                            dtype='f8',chunks=(1,128,128,2,2),compression='lzf',shuffle=True)
                    ds[j]=v
                    if j%20==0:print(f'Pack {name}: {j+1}/96 source frames (lossless float64)',flush=True)
                out.attrs['start_ms']=cfg['start'];out.attrs['dt_s']=.001
            tmp.replace(raw);dump(sig,signature)
            audit['pack_seconds']=time.perf_counter()-started
        audit.update(case=name,U=U,anchor_symmetry_max_m=symmetry,tension_formula_max_N=force_error,
                     tension_snapshot_relative_to_2T=force_error/c['tension_twice_N'],
                     tension_snapshot_note='saved force is final microstep midpoint; geometry label is saved endpoint',
                     raw_GiB=raw.stat().st_size/1024**3,source_duration_s=float(times[-1]),
                     metadata_duration_s=physics['duration_s'])
        dump(target/'audit.json',audit);reports.append(audit)
        print('Pack complete:',name,audit['counts'],flush=True)
    total=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())
    dump(dest/'manifest.json',dict(cases=reports,bytes=total,GiB=total/1024**3,
        code_hashes={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.suffix in ('.py','.json','.md') and 'data' not in p.parts and p.name!='manifest.json'},
        native_observation_dt_s=.001,source_fields='lossless float64; periodic duplicated edge removed',
        test_already_explored=True,full_run_completed=False))
    print(f'Self-contained package ready: {total/1024**3:.2f} GiB',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args();build(a.source.resolve(),a.dest.resolve())


if __name__=='__main__':main()
