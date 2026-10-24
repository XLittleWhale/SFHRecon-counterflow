"""Same cold/first-window-seeded continuation for every periodic and CF case."""
from .common import *
import argparse

def fit(c,path,net,cal,multiplier=1.,seconds=300.,seed=None):
    if (path/'summary.json').exists():return
    op=rt.make(rt.load(c['operator_source']),'cpu');train=parts(c,2);valid=parts(c,1)
    counts={k:{str(g):sum(len(z['pid'])//c['observation_steps'] for kk,z in parts(c,g) if kk==k) for g in (0,1,2)} for k in ('cap','free')}
    assert sum(v['1'] for v in counts.values()) and sum(v['2'] for v in counts.values())
    z=torch.nn.Parameter(torch.zeros_like(c['initial']) if seed is None else seed.clone())
    orders=POLICY['orders'] if seed is None else [12]
    steps=[int(n*multiplier) for n in POLICY['steps']] if seed is None else [POLICY['refine_steps']]
    lrs=POLICY['learning_rates'] if seed is None else [.0015]
    trace=[];candidates=[];stages=[];begin=time.perf_counter();updates=0;reason='step_cap'
    for ii,(K,n,lr) in enumerate(zip(orders,steps,lrs)):
        mask=controls.control_mask(c,K);opt=torch.optim.Adam([z],lr=lr);best=math.inf;bestz=z.detach().clone();before=updates
        for j in range(n+1):
            timed=time.perf_counter()-begin>seconds*sum(steps[:ii+1])/sum(steps) and j>0
            terminal=j==n or timed
            opt.zero_grad(set_to_none=True);a,_=rt.forward(c,op,z*mask)
            leaf=a.detach().requires_grad_(not terminal);obs,_=observation(c,leaf,train,net,cal,not terminal)
            reg=rt.penalties(c,z*mask,a,None,0);pen=reg['physical']+reg['spectral']
            objective=.08*obs+float(pen.detach());assert math.isfinite(objective)
            val=None;gn=None
            if j%10==0 or terminal:
                val,blocks,values=validation(c,a,valid,net,cal)
                if val<best:best=val;bestz=(z*mask).detach().clone();bestj=j
                # All candidates are K12 forward dynamics, regardless of active controls.
                candidates.append(dict(step=updates,K=K,validation=val,blocks=blocks,objective=objective,
                    theta=(z*mask).detach().clone(),scores=values))
            if not terminal:
                ((a*leaf.grad.detach()).sum()+pen).backward()
                gn=float(torch.nn.utils.clip_grad_norm_([z],10.,error_if_nonfinite=True))
                opt.param_groups[0]['lr']=lr*(1-.8*j/max(n,1));opt.step();updates+=1
            trace.append(dict(step=updates,stage_K=K,stage_step=j,objective=objective,nll=obs,validation=val,gradient_norm=gn))
            if terminal:
                if timed:reason='time_cap'
                break
        with torch.no_grad():z.copy_(bestz)
        stages.append(dict(K=K,updates=updates-before,selected_stage_step=bestj,validation=best))
        print(c['name'],path.name,'K',K,'updates',updates-before,'val',round(best,5),'seconds',round(time.perf_counter()-begin,1),flush=True)
    # Preserve original JointFull final K12-stage selection rule.
    selected=min([r for r in candidates if r['K']==12],key=lambda r:r['validation'])
    clean=lambda r:{k:v for k,v in r.items() if k not in ('theta','scores')}
    row=dict(case=c['name'],steps=updates,planned_steps=sum(steps),selected=clean(selected),stages=stages,
        seconds=time.perf_counter()-begin,stop=reason,converged=False,counts=counts,
        field_truth_used=False,test_used=False,all_data_refit=False,initial_controls=z.numel(),
        intervals=c['intervals'],inferred='single shared initial vn/vs only',**fingerprints())
    path.mkdir(parents=True,exist_ok=True);controls.archive(path,c,op,selected['theta'],row)
    rt.save(path/'validation_scores.pt',selected['scores']);rt.dump(path/'trace.json',trace)
    rt.dump(path/'candidates.json',[clean(r) for r in candidates]);gc.collect()

def choose(label,c):
    root=HERE/'fits'/label
    cold=readjson(root/'joint/summary.json');warm=readjson(root/'joint_seeded/summary.json')
    vc=rt.load(root/'joint/validation_scores.pt');vw=rt.load(root/'joint_seeded/validation_scores.pt');valid=parts(c,1);guards={}
    for kind in vc:
        ids=torch.cat([z['pid'][::c['observation_steps']]%c['pid_stride'] for k,z in valid if k==kind]);d=vw[kind]-vc[kind]
        d=torch.stack([d[ids==p].mean() for p in ids.unique()]);se=float(d.std()/math.sqrt(len(d))) if len(d)>1 else 0.
        guards[kind]=dict(physical_particles=len(d),mean_difference=float(d.mean()),paired_SE=se,pass_=float(d.mean())<=se+1e-10)
    promote=warm['selected']['validation']<cold['selected']['validation'] and all(r['pass_'] for r in guards.values())
    rt.dump(root/'selected_joint.json',dict(chosen='joint_seeded' if promote else 'joint',guards=guards,
        cold_score=cold['selected']['validation'],seeded_score=warm['selected']['validation'],field_truth_used=False,test_used=False,**fingerprints()))
    print('SELECT',label,'joint_seeded' if promote else 'joint',flush=True)

def main(args):
    declare();net,cal=model()
    for label in args.case:
        c,windows=bundle(label);root=HERE/'fits'/label
        if (root/'selected_joint.json').exists():continue
        fit(c,root/'joint',net,cal,seconds=POLICY['joint_seconds'])
        mult=c['intervals']/sum(w['intervals'] for w in windows)
        fit(windows[0],root/'independent_0',net,cal,multiplier=mult,seconds=POLICY['seed_first_seconds'])
        fit(c,root/'joint_seeded',net,cal,seconds=POLICY['refine_seconds'],seed=rt.load(root/'independent_0/selected.pt'))
        choose(label,c);gc.collect()
    paths=[HERE/'fits'/label/'selected_joint.json' for label in CASES]
    if all(p.exists() for p in paths):rt.dump(HERE/'frozen_selections.json',{p.relative_to(HERE).as_posix():rt.sha(p) for p in paths})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',nargs='+',default=CASES);main(p.parse_args())
