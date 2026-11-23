"""Non-target supervised transition fitting, selected by causal endpoint prediction."""
from common import *
import argparse
import copy
from datetime import datetime,timezone
import time
import numpy as np
import data as D
import latent_model as LM
from scoring import sequence_scores

SETTINGS=dict(epochs=80,minimum_epochs=40,patience=20,batch=1024,lr=1e-4,
              weight_decay=1e-4,gradient_clip=5.,validation_degree=20,
              selection_min_delta=1e-5,threads=2,training_dtype='float32',
              validation_dtype='float64',training_loss='supervised calibrated latent transition NLL only')

def verify_protocol():
    p=read(OUT/'protocol.json')
    b={k:v for k,v in p.items() if k!='fingerprint'}
    assert fingerprint(b)==p['fingerprint'] and p['frozen']
    for kind in ('sources','inputs'):
        for path,digest in p[kind].items():
            if sha(path)!=digest: raise RuntimeError('Frozen dependency changed: '+path)
    assert p['training_settings']==SETTINGS
    return p

def new_head(seed,dtype=torch.float32,state=None):
    torch.manual_seed(seed)
    base,record=base_network(seed,dtype)
    head=LM.LatentTransitionHead(base,state=state).to(dtype=dtype)
    return base,head,record

@torch.no_grad()
def validation(base,head,views,cal,force,degree=20):
    # Re-create rather than deepcopy: head's base reference must share the same dtype.
    b=copy.deepcopy(base).double().eval()
    h=LM.LatentTransitionHead(b,state=head.state_dict()).double().eval()
    value=mass=0.
    for item in views:
        z=item['z'];w=z['weight'].double()*2
        jets=lambda zz,ix:(z['v'][ix].double(),z['g'][ix].double(),z['acc'][ix].double())
        v,m,_=sequence_scores(z,item['c'],b,h,cal,force,w,jets,degree,'temporal')
        value+=v;mass+=m
    return value/mass

def plain_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def run(seed,max_new_epochs=None):
    p=verify_protocol();setup()
    train=D.load_views('train');val=D.load_views('validation')
    cal,force=[read(T.SOURCE/name) for name in ('calibration.json','force.json')]
    base,head,base_record=new_head(seed)
    base_before=plain_state(base)
    opt=torch.optim.AdamW(head.parameters(),lr=SETTINGS['lr'],weight_decay=SETTINGS['weight_decay'])
    pairsets=[]
    for item in train:
        previous,current,weight=D.pairs(item)
        pairsets.append((item,previous,current,weight))
    folder=OUT/'training'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    epoch=0;cpu=wall=0.;trace=[];last_improvement=0
    gen=torch.Generator().manual_seed(seed)
    if (folder/'last.pt').exists():
        last=load(folder/'last.pt')
        assert last['fingerprint']==p['fingerprint'] and last['base_sha256']==base_record['sha256']
        head.load_state_dict(last['state']);opt.load_state_dict(last['optimizer'])
        epoch,selected,trace,cpu,wall,last_improvement=[last[k] for k in
            ('epoch','selected','trace','cpu_seconds','wall_seconds','last_improvement')]
        gen.set_state(last['generator_state'])
    else:
        c,w=time.process_time(),time.perf_counter()
        v=validation(base,head,val,cal,force)
        selected=dict(state=plain_state(head),epoch=0,validation=v,seed=seed,
                      fingerprint=p['fingerprint'],base_sha256=base_record['sha256'])
        trace=[dict(epoch=0,validation=v,selected=True)]
        cpu+=time.process_time()-c;wall+=time.perf_counter()-w
    start=epoch
    complete=epoch==SETTINGS['epochs'] or (epoch>=SETTINGS['minimum_epochs'] and epoch-last_improvement>=SETTINGS['patience'])
    stopping='maximum_epochs' if epoch==SETTINGS['epochs'] else ('patience' if complete else 'running')
    def persist():
        assert all(torch.equal(v,base_before[k]) for k,v in base.state_dict().items())
        save(folder/'selected.pt',selected)
        save(folder/'last.pt',dict(state=plain_state(head),optimizer=opt.state_dict(),epoch=epoch,
            selected=selected,trace=trace,cpu_seconds=cpu,wall_seconds=wall,last_improvement=last_improvement,
            generator_state=gen.get_state(),seed=seed,fingerprint=p['fingerprint'],base_sha256=base_record['sha256']))
        dump(folder/'history.json',trace)
        dump(folder/'status.json',dict(complete=complete,seed=seed,last_epoch=epoch,
            selected_epoch=selected['epoch'],validation=selected['validation'],stopping=stopping,
            selected_sha256=sha(folder/'selected.pt'),base_sha256=base_record['sha256'],
            cpu_seconds=cpu,wall_seconds=wall,fingerprint=p['fingerprint']))
    persist()
    if complete:return selected
    while epoch<SETTINGS['epochs']:
        c,w=time.process_time(),time.perf_counter();head.train()
        total_loss=0.
        # Equal case/PID pair weights; exactly one traversal of every allowed edge.
        batches=[]
        for j,(_,prev,cur,weight) in enumerate(pairsets):
            batches.extend((j,ii) for ii in torch.randperm(len(cur),generator=gen).split(SETTINGS['batch']))
        for batch_id in torch.randperm(len(batches),generator=gen).tolist():
                j,ii=batches[batch_id]
                item,prev,cur,weight=pairsets[j];z=item['z']
                pi,ci=prev[ii],cur[ii]
                opt.zero_grad(set_to_none=True)
                loss,_=LM.supervised_pair_nll(base,head,z['h'][pi].float(),z['context'][pi].float(),
                    z['y'][pi].float(),z['q'][pi],z['h'][ci].float(),z['context'][ci].float(),
                    z['y'][ci].float(),z['q'][ci],cal)
                # Uniformly sampled batch from the complete epoch partition.
                # One global scale keeps the epoch's edge contributions equal-case/PID.
                weighted=(weight[ii].float()*loss).sum()
                objective=weighted*len(batches)
                if not torch.isfinite(objective): raise FloatingPointError('Nonfinite transition training')
                objective.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),SETTINGS['gradient_clip'],error_if_nonfinite=True)
                opt.step();total_loss+=float(weighted.detach())
        epoch+=1;head.eval()
        v=validation(base,head,val,cal,force)
        chosen=v<selected['validation']-SETTINGS['selection_min_delta']
        if chosen:
            selected=dict(state=plain_state(head),epoch=epoch,validation=v,seed=seed,
                          fingerprint=p['fingerprint'],base_sha256=base_record['sha256'])
            last_improvement=epoch
        cpu+=time.process_time()-c;wall+=time.perf_counter()-w
        trace.append(dict(epoch=epoch,training_latent_nll=total_loss,validation=v,selected=chosen,
                          cpu_seconds=cpu,wall_seconds=wall))
        complete=epoch==SETTINGS['epochs'] or (epoch>=SETTINGS['minimum_epochs'] and epoch-last_improvement>=SETTINGS['patience'])
        stopping='maximum_epochs' if epoch==SETTINGS['epochs'] else ('patience' if complete else 'running')
        persist();print('TRAIN',seed,epoch,'Vcap',v,'selected',selected['epoch'],flush=True)
        if complete or (max_new_epochs is not None and epoch-start>=max_new_epochs): break
    return selected

def freeze():
    p=verify_protocol();records=[]
    for seed in TRAIN_SEEDS:
        folder=OUT/'training'/f'seed_{seed}';status=read(folder/'status.json');ck=load(folder/'selected.pt')
        assert status['complete'] and status['fingerprint']==ck['fingerprint']==p['fingerprint']
        assert status['selected_epoch']==ck['epoch'] and sha(folder/'selected.pt')==status['selected_sha256']
        trace=read(folder/'history.json');best=trace[0]
        for row in trace[1:]:
            if row['validation']<best['validation']-SETTINGS['selection_min_delta']:best=row
        assert best['epoch']==ck['epoch']
        records.append(dict(seed=seed,path=str(folder/'selected.pt'),sha256=sha(folder/'selected.pt'),
            selected_epoch=ck['epoch'],base_sha256=ck['base_sha256'],status_path=str(folder/'status.json'),
            status_sha256=sha(folder/'status.json')))
    gate=dict(all_required_selections_frozen=True,protocol_fingerprint=p['fingerprint'],runs=records)
    gate['fingerprint']=fingerprint(gate);dump(OUT/'training/selection_frozen.json',gate)
    return gate

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','freeze'])
    parser.add_argument('--seed',type=int,choices=TRAIN_SEEDS,default=TRAIN_SEEDS[0]);parser.add_argument('--max-new-epochs',type=int)
    args=parser.parse_args();setup()
    if args.action=='run':run(args.seed,args.max_new_epochs)
    else:freeze()
