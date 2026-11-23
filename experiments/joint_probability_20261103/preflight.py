"""Full-window integration checks without H or target reference scoring."""
import copy
import time
import torch
import experiment as E

def optimizer(model):
    groups=model.parameter_groups()
    for g in groups:g['lr']={'theta':.0003,'normal':.009,'super':.009}[g['name']]
    return torch.optim.Adam(groups)

def main():
    E.M.R.setup('cpu')
    result={'heldout_scored':False,'reference_read':False,'cases':{}}
    replay,_=E.NF.prepare_replay(E.EW.RUN)
    for U in E.CASES:
        d,cfg,model,net,cal,force=E.make_context(U,E.SEEDS[0],'new_joint')
        original=E.plain_state(model);original_net=E.plain_state(net)
        started=time.perf_counter()
        first=E.field_step(model,optimizer(model),d,cfg,net,cal,force)
        seconds=time.perf_counter()-started
        updated=E.plain_state(model)
        assert all(torch.equal(v,net.state_dict()[k]) for k,v in original_net.items())
        # Poison all F responses. They must have exactly zero influence on I's update.
        altered={**d,'z':{**d['z'],'target':d['z']['target'].clone()}}
        altered['z']['target'][E.group_mask(d['z']['pid'],'F')]+=100.
        model.load_state_dict(original)
        second=E.field_step(model,optimizer(model),altered,cfg,net,cal,force)
        assert first==second
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in updated.items())
        with torch.no_grad():
            a,es,en=model()
            assert all(torch.isfinite(t).all() for t in (a,es,en))
            v=E.score(a,es,en,d,net,cal,force,'V',20,metrics=True)
        teacher=copy.deepcopy(net);heads=E.NF.configure_heads(net)
        started=time.perf_counter()
        fine=E.NF.network_step(a,es,en,d,cal,force,net,teacher,
                              torch.optim.Adam(heads['params'],lr=1e-4),replay)
        fine_seconds=time.perf_counter()-started
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in updated.items())
        ck=E.snapshot(model,net,d,cal,force,U,E.SEEDS[0],'new_joint',1,'preflight')
        model.load_state_dict(ck['model_state'])
        with torch.no_grad():
            assert all(torch.equal(x,y) for x,y in zip(model(),(ck['a'],ck['eta_s'],ck['eta_n'])))
        result['cases'][E.case_name(U)]={'full_window_frames':len(a),'full_forward_finite':True,
            'field_step_network_unchanged':True,'F_response_poison_field_update_identical':True,
            'network_step_model_unchanged':True,'snapshot_replay_exact':True,
            'field_seconds':seconds,'fine_seconds':fine_seconds,'field_step':first,
            'fine_step':fine,'validation':{k:x for k,x in v.items() if k!='per_pid'}}
        print(E.case_name(U),'passed',seconds,fine_seconds,flush=True)
        E.dump(E.OUT/'preflight/integration.json',result)
    result['passed']=True
    E.dump(E.OUT/'preflight/integration.json',result)

if __name__=='__main__':main()
