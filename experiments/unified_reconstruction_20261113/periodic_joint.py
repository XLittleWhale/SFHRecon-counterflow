"""Periodic adapter for the same controller; preserve historical G.T@u.

Full periodic data is optional and remains in its non-target source tree. The
bundle carries a small real non-target fixture for complete joint regression.
No CF boundary, clock or material-acceleration convention is applied here.
"""
import math
from types import SimpleNamespace
import numpy as np
from runtime import *
import baseline_api as A
import native_losses as N

MODEL=module('unified_periodic_closure',A.SOURCES['closure'])
ClosureNet=MODEL.ClosureNet
core=P.core
SOURCE=BASE/'outputs/FullWindowCalibration_20261026/fullwindow_4mm_results/prepared'
OPERATOR=BASE/'outputs/FullWindowCalibration_20261026/server_package/operator.pt'
ALLOWED=('vn0_0005m_s','vn0_001m_s','vn0_002m_s','vn0_0025m_s','vn0_003m_s','vn0_005m_s','vn0_0085m_s','vn0_0065m_s')


def _native(z):
    zz=dict(z);n=len(z['pid']);frame=z['frame'].double()
    zz.update(x=z['x'][:,None],h=z['h'][:,None],time=frame[:,None]+.5,
        dt=frame.new_full((n,1),.001),start=frame,end=frame+1,row_id=torch.arange(n,device=frame.device))
    # Segments reset on gaps/state switches through sequence_packs as well.
    zz['segment']=z['pid'].clone()
    return zz


def _old(z):
    return dict(z,x=z['x'][:,0],h=z['h'][:,0])


def load_case(name,device='cpu',fixture=False):
    if name not in ALLOWED:raise ValueError('Only declared non-target periodic references are supported')
    if fixture:
        d=load(HERE/'periodic_joint_fixture.pt');c=d['c'];a=d['a'];data=d['data']
    else:
        folder=SOURCE/name;data=load(folder/'training.pt');c=load(folder/'observations.pt')['c'];a=load(folder/'reference.pt')['a']
    z=data['z'];ids=torch.where(z['pid']%5!=0)[0];z=N.subset(z,ids)
    operator=MODEL.CURRENT.Momentum(core.operator(c,load(OPERATOR),core.config(),device),local=False,friction_factor=1.,nu_s=0.)
    cal,force=P.calibration()
    case=dict(case=name,c=c,op=operator,dt=.001,meta=dict(camera_fps=1000.,origin_s=0.),
        z=move(_native(z),device),labels=move(N.subset(data['labels'],ids),device),a=a.double().to(device),
        jets=tuple(j[ids,None].double().to(device) for j in data['jets']),
        force_labels=data['force'][ids,None].double().to(device),cal=cal,force=force,
        roles=dict(F=lambda p:p%5>=2,V=lambda p:p%5==1))
    if not fixture and name!='vn0_0065m_s':
        case['validation_case']=load_case('vn0_0065m_s',device)
    return case


def split(z,group):
    p=z['pid'];permitted=p%5>=2 if group=='F' else p%5==1
    inner=(p%5>=3) if group=='F' else permitted&((p//5)%2==0)
    return dict(permitted=permitted,inner=inner,outer=permitted&~inner)


def valid_starts(case,group):
    z=case['z'];splits=split(z,group);starts=[]
    for start in range(int(z['frame'].min()),int(z['frame'].max())-3):
        m=(z['frame']>=start)&(z['frame']<start+5)
        if all(len(z['pid'][m&splits[s]&(z['state']==k)].unique())>=2 for s in ('inner','outer') for k in (0,1)):
            starts.append(start)
    if not starts:raise ValueError('No valid periodic training windows')
    return starts


def window(case,start,group='F',pid_limit=None):
    z=case['z'];m=(z['frame']>=start)&(z['frame']<start+5)&split(z,group)['permitted']
    if pid_limit is not None:
        pids=[];splits=split(z,group)
        for s in ('inner','outer'):
            for k in (0,1):pids.extend(z['pid'][m&splits[s]&(z['state']==k)].unique()[:pid_limit].tolist())
        m=m&torch.isin(z['pid'],z['pid'].new_tensor(pids))
    ids=torch.where(m)[0];zz=N.subset(z,ids);zz['frame']=zz['frame']-start
    for key in ('start','end','time'):zz[key]=zz[key]-start
    basis=core.basis(zz['x'][:,0],case['c'],12)
    return dict(case=case['case'],c=case['c'],op=case['op'],dt=.001,cal=case['cal'],force=case['force'],
        z=zz,labels=N.subset(case['labels'],ids),oldjets=tuple(j[ids,0] for j in case['jets']),
        force_labels=case['force_labels'][ids,0],truth=case['a'][start:start+6],masks=split(zz,group),
        basis=basis,start=start,length=5)


_data_ns=dict(torch=torch,core=core,batched_step=MODEL.batched_step)
A.functions(A.SOURCES['C'].parent/'data_training.py',['trajectory','reference_residual','query','relabel_force'],_data_ns)
trajectory=_data_ns['trajectory'];reference_residual=_data_ns['reference_residual']


def query(w,a,residual=None):
    return tuple(j[:,None] for j in _data_ns['query'](dict(w,z=_old(w['z'])),a,residual))


def relabel_force(w,jets):
    return _data_ns['relabel_force'](dict(w,z=_old(w['z'])),tuple(j[:,0] for j in jets))[:,None]


def initial_scale(like):
    modes,_=core.modes(12);ring=np.max(np.abs(modes),1).repeat(2)
    return 1/(1+(like.new_tensor(ring)/4)**2)


def kernels(w):
    data=SimpleNamespace(trajectory=trajectory,query=query,reference_residual=reference_residual,relabel_force=relabel_force)
    prob=SimpleNamespace(core=core,window_nll=N.window_nll,supervised_anchors=N.supervised_anchors)
    return A.joint_kernels(data,prob,initial_scale)


def legacy_kernels():
    ns=dict(torch=torch,math=math,LM=A.LM,TM=A.TM,core=core,ob=P._runtime.ob)
    A.functions(A.SOURCES['joint_probability'],['step','pid_weights','sequence_layout','window_nll','supervised_anchors'],ns)
    return A.joint_kernels(SimpleNamespace(**{k:_data_ns[k] for k in ('trajectory','reference_residual','query','relabel_force')}),
        SimpleNamespace(core=core,window_nll=ns['window_nll'],supervised_anchors=ns['supervised_anchors']),initial_scale)


def legacy_window(w):return dict(w,z=_old(w['z']))


def closed(case,net):return case['op'] if net is None else MODEL.ClosureMomentum(case['op'],net)


@torch.no_grad()
def stability(case,op,settings):
    def run(operator):
        a=case['a'][0];peak=a.square().sum(-1).sqrt()
        for _ in range(len(case['a'])-1):
            a=MODEL.batched_step(operator,a,settings['substeps'])
            if not bool(torch.isfinite(a).all()) or float(a.abs().max())>settings['full_window_amplitude_limit']:
                return dict(passed=False,reason='nonfinite_or_amplitude')
            peak=torch.maximum(peak,a.square().sum(-1).sqrt())
        return dict(passed=True,peak=peak.cpu().tolist(),frames=len(case['a']))
    if '_stability' not in case:case['_stability']=run(case['op'])
    baseline=case['_stability'];result=run(op)
    if not baseline['passed']:raise RuntimeError('Periodic physical baseline unstable')
    if result['passed']:
        ratio=np.array(result['peak'])/np.maximum(baseline['peak'],1e-12)
        result.update(peak_growth_ratio=ratio.tolist(),passed=bool((ratio<=settings['full_window_growth_factor']).all()))
    return result
