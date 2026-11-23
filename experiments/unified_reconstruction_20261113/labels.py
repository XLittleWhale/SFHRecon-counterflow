"""Recover historical native-start labels without replay or guessed signs.

Valid only for the exact hash-certified Galantucci fixed-slot source version.
The initial checkpoint is authoritative; the final checkpoint is not used.
"""
import numpy as np
from runtime import *

def sign_labels(sign_table, attached, state):
    sign_table=np.asarray(sign_table)
    if sign_table.ndim!=1 or not np.isin(sign_table,[-1,1]).all():
        raise ValueError('Require the actual +/-1 initial circulation table')
    cap=state==1
    if not np.isin(state,[0,1]).all() or not np.array_equal(cap,attached>=0):
        raise ValueError('State/attachment mismatch')
    if (attached[cap]>=len(sign_table)).any():raise ValueError('Attachment index out of bounds')
    q=np.full(state.shape,-1,np.int64)
    q[cap]=(sign_table[attached[cap]]>0).astype(np.int64)
    return q

def geometry_labels(anchors,particle,state,width,height,radius):
    delta=np.asarray(anchors,np.float64)-np.asarray(particle,np.float64)[...,None,:]
    # x has walls; only y is periodic. No x minimum-image operation is valid.
    delta[...,1]-=height*np.round(delta[...,1]/height)
    cap=state==1
    symmetry=float(np.max(np.abs(delta[...,0,:][cap]-delta[...,1,:][cap]),initial=0))
    if symmetry>1e-10:raise ValueError('Two anchors cannot be represented by the single-anchor baseline')
    y=delta[...,0,:]/radius
    y[~cap]=np.nan
    return y,symmetry

def verify_source(source,solver):
    source=Path(source);solver=Path(solver)
    certificate=read(HERE/'fixed_slot_certificate.json')
    c=read(source/'configuration.json');m=read(source/'particles/metadata.json')
    if c['version']!=certificate['version'] or c['implementation_sha256']!=certificate['implementation_sha256']:
        raise RuntimeError('No fixed-slot proof for this solver implementation')
    for name,digest in certificate['implementation_sha256'].items():
        if sha(solver/name)!=digest:raise RuntimeError('Certified solver file changed: '+name)
    start=source/'spinup/particle_start_state.npz'
    if sha(start)!=m['source_sha256']:raise RuntimeError('Particle run is not bound to this initial checkpoint')
    for name in ('particle_trajectories.npz','particle_hidden_truth.npz','frame_manifest.json'):
        if sha(source/'particles'/name)!=m['output_sha256'][name]:raise RuntimeError('Raw particle file changed: '+name)
    with np.load(start,allow_pickle=False) as ck:
        cfg=json.loads(str(ck['config_json']));progress=json.loads(str(ck['runner_state_json']))
        if cfg!=c['config'] or progress['configuration_sha256']!=c['configuration_sha256'] or not progress['complete']:
            raise RuntimeError('Initial checkpoint configuration/commit mismatch')
        if abs(float(ck['time_s'])-c['spinup']['actual_duration_s'])>1e-10:
            raise RuntimeError('Not the particle seeding time')
        signs=ck['vortex_sign'].copy()
    return c,signs,dict(initial_checkpoint_sha256=sha(start),
        certificate_sha256=sha(HERE/'fixed_slot_certificate.json'),
        sign_source='actual particle-start checkpoint + hash-certified immutable array slots',
        final_checkpoint_read=False,parity_guess_used=False,local_vorticity_used=False,
        raw_sha256={name:sha(source/'particles'/name) for name in ('particle_trajectories.npz','particle_hidden_truth.npz')})

def prepare(source,prepared,solver,output):
    source,prepared,output=map(Path,(source,prepared,output))
    if output.resolve().is_relative_to(source.resolve()) or output.resolve().is_relative_to(prepared.resolve()):
        raise ValueError('Write labels to a separate output; original results are read-only')
    c,signs,evidence=verify_source(source,solver)
    obs=load(prepared/'probability_observations.pt');z=obs['z']
    if not bool(((z['pid']%5==2)|(z['pid']%5==1)).all()):raise ValueError('Only F/V may enter label preparation')
    meta=obs['meta']
    if meta['source_sha256']['configuration']!=sha(source/'configuration.json'):
        raise RuntimeError('Prepared observations belong to a different source')
    for key,name in (('observations','particle_trajectories.npz'),('hidden_labels','particle_hidden_truth.npz')):
        if meta['source_sha256'][key]!=evidence['raw_sha256'][name]:raise RuntimeError('Prepared raw-data binding mismatch')
    with np.load(source/'particles/particle_trajectories.npz',allow_pickle=False) as d:
        times=d['time_s'];pid=d['particle_id'];xy=d['xy_unwrapped_m']
    with np.load(source/'particles/particle_hidden_truth.npz',allow_pickle=False) as d:
        if not np.array_equal(times,d['time_s']) or not np.array_equal(pid,d['particle_id']):raise RuntimeError('Hidden/observed clocks or IDs differ')
        state=d['state'];attached=d['attached_vortex'];anchors=d['anchors_unwrapped']
    q=sign_labels(signs,attached,state)
    p=c['config'];y,symmetry=geometry_labels(anchors,xy,state,p['width_m'],p['height_m'],p['particle_radius_m'])
    lookup={int(v):i for i,v in enumerate(pid)}
    frame=z['frame'].numpy();columns=np.array([lookup[int(v)] for v in z['pid']])
    if not np.array_equal(state[frame,columns],z['state'].numpy()):raise RuntimeError('Observation row state mismatch')
    labels=dict(y=torch.from_numpy(y[frame,columns]),q=torch.from_numpy(q[frame,columns]),
        time_s=torch.from_numpy(times[frame]),row_id=z['row_id'].clone())
    evidence.update(observation_sha256=sha(prepared/'probability_observations.pt'),
        configuration_sha256=sha(source/'configuration.json'),anchor_symmetry_max_m=symmetry,
        label_time='native interval start, as in original periodic build.py',
        y_units='anchor0 minus particle, y-only minimum image, divided by radius',
        q_encoding='0:negative, 1:positive; -1:free and not a latent target',
        synthetic_substep_latent_labels=False,rows=len(frame),captured_rows=int((z['state']==1).sum()))
    save(output/'latent_labels.pt',dict(**labels,meta=evidence))
    dump(output/'label_provenance.json',evidence)
    return evidence
