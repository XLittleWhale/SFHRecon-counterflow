"""Optional data-backed checks; skipped in a source-only checkout."""
from pathlib import Path
import importlib
import json
import numpy as np
import pytest
import torch
from counterflow.paths import DATA

pytestmark=pytest.mark.skipif(not(DATA/'inputs/periodic_4.pt').exists(),reason='Portable research data bundle not installed')

def test_training_objective_has_finite_gradient():
    from counterflow.common import ASSETS,rt,Density
    from counterflow.training_objective import score
    tr,_=rt.load(DATA/'training/views.pt')
    z={k:v[:8].clone() for k,v in tr[3].items()}
    net=Density();net.load_state_dict(rt.load(ASSETS/'seed.pt')['state'])
    opt=torch.optim.AdamW(net.parameters(),lr=3e-5)
    loss,_,_=score(net,z,8);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    opt.step()

@pytest.mark.parametrize('label',['periodic_4','cf_261'])
def test_inverse_gradient_and_short_fit(label,tmp_path):
    from counterflow import common as cm
    inv=importlib.import_module('counterflow.invert')
    cm.rt.setup('cpu');net,cal=cm.model();c,_=cm.bundle(label)
    op=cm.rt.make(cm.rt.load(c['operator_source']),'cpu')
    torch.manual_seed(17);theta=(torch.randn_like(c['initial'])*1e-5).requires_grad_()
    batches=cm.parts(c,2);small=[]
    for kind in ('cap','free'):
        first=next((z for k,z in batches if k==kind),None)
        if first is not None:small.append((kind,{k:v[:c['observation_steps']] for k,v in first.items()}))
    a,_=cm.rt.forward(c,op,theta);leaf=a.detach().requires_grad_()
    cm.observation(c,leaf,small,net,cal,True,degree=8)
    (a*leaf.grad).sum().backward();direction=torch.randn_like(theta);direction/=direction.norm()
    ad=float((theta.grad*direction).sum());v=[];eps=1e-5
    with torch.no_grad():
        for sign in (-1,1):
            trial,_=cm.rt.forward(c,op,theta+sign*eps*direction)
            value,_=cm.observation(c,trial,small,net,cal,degree=8);v.append(.08*value)
    fd=(v[1]-v[0])/(2*eps)
    assert abs(ad-fd)<1e-6+1e-4*abs(ad)
    # Reversible smoke fit: identical algorithm, one update in each continuation stage.
    saved=inv.POLICY.copy()
    try:
        inv.POLICY.update(steps=[1,1,1,1])
        inv.fit(c,tmp_path/label,net,cal,seconds=60)
    finally:
        inv.POLICY.clear();inv.POLICY.update(saved)
    meta=json.loads((tmp_path/label/'summary.json').read_text())
    assert meta['steps']==4 and not meta['field_truth_used'] and not meta['test_used']
    assert (tmp_path/label/'selected.pt').exists()
