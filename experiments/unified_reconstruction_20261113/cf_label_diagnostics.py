"""Discarded CF supervised-label probes, not a substitute for stages A/B/C.

Only the original marginal label terms are tested. Native endpoint and latent
transition losses are deliberately not approximated by a 1 ms observation.
No optimizer/checkpoint produced here is eligible for production selection.
"""
import time
from runtime import *
import baseline_api as baseline


def directory(case):
    packaged = BASE/'prepared_supervision'/case
    local = BASE/'outputs/UnifiedReconstruction_20261113_GPU/validation/existing_data'/case
    return packaged if packaged.is_dir() else local


def probe(case, device='cpu', steps=8):
    folder=directory(case)
    path=BASE/'outputs/CFSimEachPINN_20261109/prepared'/case/'probability_observations.pt'
    obs=load(path); labels=load(folder/'latent_labels.pt'); reference=load(folder/'full_resolved_context.pt')
    if labels['meta']['observation_sha256']!=sha(path):
        raise RuntimeError('CF diagnostic observation provenance changed')
    if reference['meta']['latent_label_sha256']!=sha(folder/'latent_labels.pt'):
        raise RuntimeError('CF diagnostic label provenance changed')
    z=obs['z']; c=reference['c']; output={}
    for code,kind in ((0,'free'),(1,'cap')):
        allowed=(z['pid']%5==2)&(z['state']==code)
        # Eight different permitted physical particles, one native row each.
        # This tiny balanced probe measures learnability, not validation skill.
        available=torch.where(allowed)[0]
        chosen=[]
        for pid in z['pid'][available].unique(sorted=True)[:8]:
            chosen.append(int(torch.where(allowed&(z['pid']==pid))[0][0]))
        if len(chosen)<2:raise RuntimeError('Insufficient F particles for label diagnostic')
        ids=torch.tensor(chosen,dtype=torch.long)
        net,cal,force,origin=B.load(301030,device)
        baseline.configure_stage(net,'A'); net.double()
        before_head={k:v.detach().clone() for k,v in net.head.state_dict().items()}
        before_buffers={k:v.detach().clone() for k,v in net.named_buffers()}
        h=z['h'][ids,0].to(device)
        v,g=(reference['resolved_jets'][j][ids,0].to(device) for j in (0,1))
        ctx=P.core.context(c,dict(w=z['w'][ids].to(device)),v,g)
        y,q=labels['y'][ids].to(device),labels['q'][ids].to(device)
        target=reference['free_force'][ids,0].to(device)
        def objective():
            if code==0:
                return baseline.TM.free_force_nll(net.base,h,ctx,target,force).mean()
            return -net.base.log_prob(h,ctx,y,q).mean()
        optimizer=torch.optim.Adam(net.base.parameters(),lr=1e-4)
        losses=[]; norms=[]; start=time.perf_counter()
        for step in range(steps+1):
            optimizer.zero_grad(set_to_none=True)
            loss=objective()
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite CF supervised label NLL')
            losses.append(float(loss.detach()))
            if step==steps:break
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(net.base.parameters(),5.,error_if_nonfinite=True)
            norms.append(float(norm))
            if not norms[-1]>0:raise RuntimeError('Zero CF supervised gradient')
            optimizer.step()
        if not losses[-1]<losses[0]:raise RuntimeError('CF marginal label probe did not reduce its loss')
        if any(not torch.equal(v,before_head[k]) for k,v in net.head.state_dict().items()):
            raise RuntimeError('Stage A diagnostic modified frozen transition parameters')
        if any(not torch.equal(v,before_buffers[k]) for k,v in net.named_buffers()):
            raise RuntimeError('Diagnostic changed inherited normalization/calibration')
        if any(p.grad is not None for p in net.head.parameters()):raise RuntimeError('Unexpected transition gradient')
        output[kind]=dict(rows=len(ids),pids=z['pid'][ids].tolist(),row_ids=z['row_id'][ids].tolist(),
            initial=losses[0],final=losses[-1],losses=losses,gradient_l2_before_clip=norms,
            updates=steps,learning_rate=1e-4,seconds=time.perf_counter()-start,
            frozen_head_unchanged=True,normalization_unchanged=True)
    return dict(case=case,device=device,dtype='float64',passed=True,components=output,
        observation_sha256=sha(path),label_sha256=sha(folder/'latent_labels.pt'),
        force_context_sha256=sha(folder/'full_resolved_context.pt'),
        role='F only; discarded small component diagnostic; no V/H selection',
        production_checkpoint_written=False,endpoint_term_tested=False,
        native_transition_tested=False,channel_joint_tested=False,
        limitation='Component learnability only; not complete stage A or formal CF method acceptance')


def run(device='cpu'):
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    return [probe(f'cf_2K_Q{q}',device) for q in (113,175,217)]
