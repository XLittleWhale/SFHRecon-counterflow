"""Load the validated coordinates, operator and tolerance without refitting."""
from functools import lru_cache
from types import SimpleNamespace
import torch
from runtime import BASE, F, P, load, move


def transfer(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,dict):return {k:transfer(v,device) for k,v in value.items()}
    if isinstance(value,list):return [transfer(v,device) for v in value]
    if isinstance(value,tuple):return tuple(transfer(v,device) for v in value)
    return value


@lru_cache(maxsize=1)
def asset():
    return load(BASE/'revision_assets/frozen_physics.pt')


class ResidualPrior:
    def __init__(self,data,device):
        self.intervals=data['intervals']
        self.normal_transform=data['normal_transform'].to(device)
        self.super_transform=data['super_transform'].to(device)
        self.normal_temporal_cholesky=data['normal_temporal_cholesky'].to(device)

    def whiten_normal(self,residual):
        if residual.shape[-2:]!=(self.intervals,self.normal_transform.shape[0]):
            raise ValueError('Complete temporal residual required')
        spatial=residual@self.normal_transform
        return torch.linalg.solve_triangular(self.normal_temporal_cholesky,
            spatial.transpose(-1,-2).unsqueeze(-1),upper=False).squeeze(-1).transpose(-1,-2)

    def whiten_super(self,residual):
        return residual@self.super_transform


def context(out,row,seed,device):
    from pathlib import Path
    import calibrated_workflow as workflow
    pack=load(Path(out)/row['path']/'inversion_observations.pt')
    data=asset();meta=pack['meta']
    if meta['intervals']!=data['intervals'] or meta['physics_config']!=data['physics_config']:
        raise RuntimeError('Frozen physics does not match the target configuration')
    z=move(pack['z'],device)
    if not bool(((z['pid']%5>=3)|(z['pid']%5==1)).all()):
        raise ValueError('Field input must contain I/V only')
    c=transfer(data['context'],device)
    op=SimpleNamespace(**transfer(data['operator'],device))
    cold=data['cold'].to(device)
    field=F.make_field(c,cold,seed,duration_s=data['intervals']*.001)
    # Exact frozen CPU initialization also fixes repeated-eigenvalue coordinates.
    field.load_state_dict(data['initial_fields'][seed],strict=True)
    net,cal,force,digest=workflow.frozen(out,row['reference'],seed,device)
    return dict(c=c,z=z,op=op,prior=ResidualPrior(data['prior'],device),field=field,
        net=net,cal=cal,force=force,digest=digest)
