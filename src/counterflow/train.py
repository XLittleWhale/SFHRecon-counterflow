"""Bounded, validation-selected retraining of ONE common model, no inverse truth."""
from .common import *

from . import training_objective as trmod

def main():
    declare()
    if (HERE/'fits').exists():raise ValueError('Training requires a run directory without existing inverse fits.')
    tr,va=rt.load(DATA/'training/views.pt')
    dest=HERE/'training'; dest.mkdir(exist_ok=True)
    config=dict(seeds=[141001,141002],lr=3e-5,batch=512,max_epochs=64,patience=12,
        max_seconds_per_seed=180,train_sobol=32,validation_sobol=64,
        init='existing H18; continuation retraining, not training from scratch',
        selection='mean validation K12/16/18; epoch zero eligible; no per-case network',
        data_sha=rt.sha(DATA/'training/views.pt'),seed_sha=rt.sha(ASSETS/'seed.pt'))
    rt.dump(dest/'policy.json',config)
    seedck=rt.load(ASSETS/'seed.pt'); N=len(tr[0]['q'])
    stack={k:torch.stack([z[k] for z in tr]) for k in tr[0]};runs=[]
    for sd in config['seeds']:
        path=dest/f'seed_{sd}'
        if (path/'summary.json').exists(): runs.append(readjson(path/'summary.json'));continue
        rt.seed(sd); net=Density();net.load_state_dict(seedck['state'])
        opt=torch.optim.AdamW(net.parameters(),lr=config['lr'],weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=4,min_lr=3e-6)
        _,per=trmod.validate(net,va,64,512);best=float(np.mean([r[0] for r in per[3:]])); initial=best
        state=rt.cpu_state(net);bestep=0;steps=0;start=time.perf_counter();trace=[dict(epoch=0,validation=best,per_order=per)];reason='max_epochs'
        for ep in range(1,65):
            net.train();choose=torch.randint(6,(N,));total=0;seen=0
            for ix in torch.randperm(N).split(512):
                if time.perf_counter()-start>180:reason='bounded_CPU_time';break
                z={k:v[choose[ix],ix] for k,v in stack.items()};opt.zero_grad(set_to_none=True)
                loss,_,_=trmod.score(net,z,32);assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step()
                total+=float(loss.detach())*len(ix);seen+=len(ix);steps+=1
            _,per=trmod.validate(net,va,64,512);v=float(np.mean([r[0] for r in per[3:]]));sched.step(v)
            if v<best:best=v;bestep=ep;state=rt.cpu_state(net)
            trace.append(dict(epoch=ep,train=total/max(seen,1),validation=v,per_order=per,lr=opt.param_groups[0]['lr'],samples=seen))
            if ep%4==0 or reason=='bounded_CPU_time': print(sd,'epoch',ep,'best',round(best,5),'selected',bestep,flush=True)
            if reason=='bounded_CPU_time':break
            if ep-bestep>=12:reason='validation_early_stop';break
        row=dict(seed=sd,selected_epoch=bestep,last_epoch=ep,steps=steps,baseline=initial,validation=best,
            seconds=time.perf_counter()-start,stop=reason,from_scratch=False,converged=False,field_output_labels=False)
        rt.save(path/'best.pt',seedck|dict(state=state));rt.dump(path/'summary.json',row);rt.dump(path/'trace.json',trace);runs.append(row)
    chosen=min(runs,key=lambda r:r['validation']);rt.save(dest/'selected.pt',rt.load(dest/f"seed_{chosen['seed']}/best.pt"))
    rt.dump(dest/'selection.json',dict(runs=runs,selected=chosen,same_model_all_cases=True))
    print('ONE GLOBAL MODEL',chosen,flush=True)

if __name__=='__main__':main()
