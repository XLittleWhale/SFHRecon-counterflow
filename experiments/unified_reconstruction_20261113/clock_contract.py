"""Geometry-independent physical-time windows and explicitly measured pairs."""
import math
import torch

def native_window_intervals(fps,minimum_seconds=.005,minimum_intervals=2):
    if fps<=0:raise ValueError('Positive observation frequency required')
    return max(minimum_intervals,math.ceil(minimum_seconds*fps-1e-10))

def split_reference(pid,group):
    rem={'F':2,'V':1}[group]
    allowed=pid%5==rem
    inner=allowed&((pid//5)%2==0)
    outer=allowed&((pid//5)%2==1)
    return dict(inner=inner,outer=outer,permitted=allowed)

def pairs(z,mask):
    lookup={(int(z['pid'][i]),int(z['frame'][i])):i for i in torch.where(mask&(z['state']==1))[0]}
    previous=[];current=[]
    for (pid,frame),i in sorted(lookup.items()):
        j=lookup.get((pid,frame+1))
        if j is not None and z['segment'][i]==z['segment'][j] and abs(float(z['end'][i]-z['start'][j]))<1e-7:
            previous.append(i);current.append(j)
    before=torch.tensor(previous,dtype=torch.long);after=torch.tensor(current,dtype=torch.long)
    return before,after,z['start'][after]-z['start'][before]

def require_one_ms(dt_ms):
    if not len(dt_ms) or not torch.allclose(dt_ms,torch.ones_like(dt_ms),rtol=0,atol=1e-7):
        raise RuntimeError('The original supervised transition kernel accepts 1 ms pairs only; native-cadence adapter not validated')

def audit_windows(z,fps):
    length=native_window_intervals(fps);report={}
    for group in ('F','V'):
        masks=split_reference(z['pid'],group);valid=[];invalid=[]
        for start in range(int(z['frame'].min()),int(z['frame'].max())-length+2):
            window=(z['frame']>=start)&(z['frame']<start+length)
            counts={f'{split}_{state}':len(z['pid'][window&masks[split]&(z['state']==state)].unique())
                for split in ('inner','outer') for state in (0,1)}
            previous,current,dt=pairs(z,window&masks['permitted'])
            item=dict(start=start,counts=counts,paired_rows=len(current))
            (valid if min(counts.values())>=2 and len(current)>0 else invalid).append(item)
        report[group]=dict(valid=valid,invalid=invalid)
    return dict(rule='max(5 ms, 2 native intervals); identical rule for every geometry',
        native_intervals=length,physical_duration_s=length/fps,synthetic_endpoints=False,groups=report)
