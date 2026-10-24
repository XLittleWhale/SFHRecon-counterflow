"""Fit latent/captured calibration; unchanged free decoder retains its calibrated covariance."""
import json,math
from pathlib import Path
import numpy as np
import torch
from .observation import *
from .io import *
from .paths import DATA,RUN,ASSETS,CONFIG
from .calibration_objective import *

def finish_correlation(path):
    current=json.loads(path.read_text())
    frozen=json.loads((ASSETS/'calibration.json').read_text())
    # Free dynamics have no neural weights. Original calibration data, algorithm
    # and free-noise optimum must still match before transferring its beta.
    manifest=json.loads((CONFIG/'data_manifest.json').read_text())
    if sha(DATA/'calibration/inputs.pt')!=manifest['calibration/inputs.pt']['sha256']:
        raise ValueError('Changed calibration dataset requires new free-correlation calibration.')
    for k in ('2','4','8','12'):
        if not np.allclose(current['sigma'][k]['free'],frozen['sigma'][k]['free'],rtol=0,atol=1e-8):
            raise ValueError('Free noise law changed; cannot reuse its correlation calibration.')
    correlation=json.loads((ASSETS/'correlation.json').read_text())
    correlation['reuse_reason']='Identical free decoder, noise law and calibration inputs; network retraining changes captured model only.'
    dump(RUN/'correlation.json',correlation)

def main():
    from types import SimpleNamespace
    args=SimpleNamespace(device="cpu",out=str(RUN),smoke=False)
    device=setup(args.device);cfg=json.loads((CONFIG/'calibration.json').read_text());out=Path(args.out);dest=out/'calibration'
    if args.smoke:cfg=cfg|dict(max_steps=3,min_steps=1,patience=1,degree=8)
    fingerprint=sha(out/'training/selected.pt');provenance=dict(checkpoint_sha256=fingerprint,input_sha256=sha(DATA/'calibration/inputs.pt'),config=cfg,smoke=args.smoke)
    if (dest/'provenance.json').exists():assert json.loads((dest/'provenance.json').read_text())==provenance,'Calibration inputs changed: choose a new output directory.'
    if (dest/'calibration.json').exists():
        finish_correlation(dest/'calibration.json');return
    dump(dest/'provenance.json',provenance);net=Density().double().to(device);net.load_state_dict(load(out/'training/selected.pt')['state']);net.eval()
    rows=load(DATA/'calibration/inputs.pt')
    if args.smoke:
        rows=[r|{k:v[:8] for k,v in r.items() if torch.is_tensor(v)} for r in rows]
    rows=move(rows,device)
    with torch.no_grad():
        for r in rows:
            e=net.encode(r['h'],r['c']);_,mu,L=net.parameters_at(e);r.update(logits=net.sign(e),mu=mu[:,:,0],L=L[:,:,0])
    tr=[r for r in rows if r['split']=='train'];va=[r for r in rows if r['split']=='validation'];opts=[]
    for kind in ('identity','global','log_speed'):
        initial=torch.zeros(3,dtype=torch.float64,device=device);mask=initial.new_tensor([1,kind=='log_speed',1])
        if kind=='identity':p,fit=initial,dict(steps=0,reason='identity_baseline')
        else:p,fit=optimize(lambda p:latent_score(tr,p)+.01*p[1].square()+.005*p[2].square(),initial,mask,cfg)
        opts.append(dict(kind=kind,parameters=p.tolist(),validation_NLL=float(latent_score(va,p)),fit=fit))
    chosen=min(opts,key=lambda r:r['validation_NLL']);p=chosen['parameters'];cal=dict(latent=p[:2],sign_temperature=float(np.clip(np.exp(p[2]),.25,4)),sigma={});reports=[]
    for K in (2,4,8,12):
        cal['sigma'][str(K)]={}
        for state in ('captured','free'):
            train=residuals(tr,cal,K,state,cfg['degree']);valid=residuals(va,cal,K,state,cfg['degree']);options=[]
            for kind in ('original','global','log_speed'):
                initial=torch.tensor([math.log(.3),0.],dtype=torch.float64,device=device);mask=initial.new_tensor([1,kind=='log_speed'])
                if kind=='original':p,fit=initial,dict(steps=0,reason='original_noise_baseline')
                else:p,fit=optimize(lambda p:noise_score(train,p)+.01*p[1].square(),initial,mask,cfg)
                options.append(dict(kind=kind,parameters=p.tolist(),validation_NLL=float(noise_score(valid,p)),fit=fit))
            selected=min(options,key=lambda r:r['validation_NLL']);cal['sigma'][str(K)][state]=selected['parameters'];reports.append(dict(K=K,state=state,candidates=options,selected=selected['kind']))
        print(f'Calibrated K{K}',flush=True)
    cal.update(checkpoint_sha256=fingerprint,field_condition_orders=[2,4,8,12],test_data_used=False,
        note='CF retains fixed previous CF noise laws, no experiment noise refit; latent means unchanged.')
    dump(dest/'selection.json',dict(latent_candidates=opts,selected_latent=chosen['kind'],noise=reports));dump(dest/'calibration.json',cal)
    finish_correlation(dest/"calibration.json")
