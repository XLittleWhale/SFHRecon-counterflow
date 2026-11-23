"""Allowed-data integration and disposable cost profile before protocol freeze."""
from common import *
import time
import copy
import numpy as np
import train_latent as N
import experiment_latent as E
import scoring

def main():
    setup();report={}
    cal,force=[read(T.SOURCE/n) for n in ('calibration.json','force.json')]
    base,head,_=N.new_head(TRAIN_SEEDS[0]);views=N.D.load_views('train');validation=N.D.load_views('validation')
    # Two disposable epochs; their weights never enter the formal runs.
    opt=torch.optim.AdamW(head.parameters(),lr=N.SETTINGS['lr'],weight_decay=N.SETTINGS['weight_decay'])
    rng=torch.Generator().manual_seed(991104)
    started=time.perf_counter();pilot=[]
    for epoch in range(2):
        pairs=[(item,*N.D.pairs(item)) for item in views];batches=[]
        for j,(_,prev,cur,weight) in enumerate(pairs):
            batches.extend((j,ix) for ix in torch.randperm(len(cur),generator=rng).split(1024))
        for bid in torch.randperm(len(batches),generator=rng).tolist():
            j,ii=batches[bid];item,prev,cur,weight=pairs[j];z=item['z'];pi,ci=prev[ii],cur[ii]
            opt.zero_grad(set_to_none=True)
            loss,diag=N.LM.supervised_pair_nll(base,head,z['h'][pi],z['context'][pi],z['y'][pi],z['q'][pi],
                z['h'][ci],z['context'][ci],z['y'][ci],z['q'][ci],cal)
            obj=(weight[ii]*loss).sum()*len(batches);obj.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
        v20=N.validation(base,head,validation,cal,force,20)
        v28=N.validation(base,head,validation,cal,force,28)
        pilot.append(dict(epoch=epoch+1,cap_validation20=v20,cap_validation28=v28,difference=v28-v20))
        print('PILOT',pilot[-1],flush=True)
    report['disposable_pilot']=dict(epochs=2,seconds=time.perf_counter()-started,values=pilot,formal_weights_reused=False)
    qweight=float(head.network[-1].weight[0].detach().abs().sum())
    assert qweight>0, 'q output must leave its zero initialization in real allowed-data training'
    qk=[]
    with torch.no_grad():
        for item in validation:
            prev,cur,ww=N.D.pairs(item);zz=item['z']
            for ii in torch.arange(len(cur)).split(1024):
                kk,*_=head.parameters_at(zz['h'][cur[ii]].float(),zz['context'][cur[ii]].float(),zz['context'][prev[ii]].float())
                qk.append(kk.double())
    qk=torch.cat(qk)
    assert float(qk.max())>0, 'real validation context must activate learned q persistence'
    report['real_q_learning_regression']=dict(q_output_weight_l1=qweight,kappa_mean=float(qk.mean()),
        kappa_max=float(qk.max()),positive_fraction=float((qk>0).double().mean()),formal_weights_reused=False)
    # I-only cold field regression against the frozen parent, no reference reads.
    d,cfg,model,net,cal,force=E.make_context(.004,SEEDS[0],'iid',initial_network=True)
    a,es,en=model();old=S.score(a,es,en,d,net.base,cal,force,'I',12)
    new=E.score(a,es,en,d,net,cal,force,'I',12)
    assert max(abs(old[k]-new[k]) for k in old)<1e-10
    older=copy.deepcopy(model)
    groups=model.parameter_groups();ogroups=older.parameter_groups()
    for gg in (groups,ogroups):
        for g in gg:g['lr']=dict(theta=.0003,super=.009,normal=.009)[g['name']]
    op=torch.optim.Adam(groups);oop=torch.optim.Adam(ogroups)
    oldstep=S.field_step(older,oop,d,cfg,net.base,cal,force)
    t=time.perf_counter();newstep=E.field_step(model,op,d,cfg,net,cal,force)
    field_delta=max(float((v-older.state_dict()[k]).abs().max()) for k,v in model.state_dict().items())
    assert field_delta<1e-10
    report['iid_regression']=dict(score_old=old,score_new=new,one_step_parameter_max_abs=field_delta,
        seconds=time.perf_counter()-t,network_frozen=all(not p.requires_grad for p in net.parameters()))
    # Nonzero trained transition, both full target I paths, no target network updates.
    for mode in ('reset','temporal'):
        d,cfg,m,n,cal,force=E.make_context(.0015,SEEDS[0],mode,initial_network=True)
        n.head.load_state_dict(head.state_dict());n.freeze();before=E.plain_state(n)
        groups=m.parameter_groups()
        for g in groups:g['lr']=dict(theta=.0003,super=.009,normal=.009)[g['name']]
        opt=torch.optim.Adam(groups);t=time.perf_counter();r=E.field_step(m,opt,d,cfg,n,cal,force)
        assert all(torch.equal(v,before[k]) for k,v in n.state_dict().items())
        report[mode+'_integration']=dict(seconds=time.perf_counter()-t,step=r,network_unchanged=True)
    # Static mask audit also covers padding, which previously used global row 0.
    for group in ('I','V'):
        mask=S.group_mask(d['z']['pid'],group)&(d['z']['state']==1)
        for pack in N.D.pack_sequences(d['z'],mask,64):
            indices,valid=pack['index'],pack['valid'];safe=torch.where(valid,indices,indices[valid][0])
            assert S.group_mask(d['z']['pid'][safe.flatten()],group).all()
    report['padding_groups']='All actual and dummy rows stay within requested group'
    report['pre_protocol_defect_disclosure']='Two early cold-field timing calls used global row 0 for zero-weight padding; it could belong to H. No H metrics were output or used for selection, weights were zero and segment resets blocked propagation. Replaced by within-group filler before all formal training/inversion and verified here. Target data were also historically explored; this is a development comparison, not fresh blind inference.'
    report['complete']=True
    report['sources']={str(p):sha(p) for p in HERE.glob('*.py')}
    dump(OUT/'preflight/integration.json',report)
    print('INTEGRATION_PREFLIGHT_COMPLETE',flush=True)

if __name__=='__main__':main()
