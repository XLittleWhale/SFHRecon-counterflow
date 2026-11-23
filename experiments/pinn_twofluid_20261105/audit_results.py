"""Read-only arithmetic, provenance and export audit after global selection.

No optimization, target observations, raw target reference.pt, or likelihood
evaluation is performed. Only frozen checkpoints and completed exported metrics
are inspected. Reconstructed probability modules are instantiated solely to
check frozen weights, never supplied observations.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[2]/'outputs/PINNTwoFluid_20261105'
CASES={'vn0_0015m_s':.0015,'vn0_004m_s':.004}
ARMS=('global_spline','pinn')
SEEDS=(301030,301031,301032)
BUDGETS=(200,600)
MODES=('primary','terminal')


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(path)


class Audit:
    def __init__(self):
        self.report=dict(version='pinn-twofluid-results-audit-v1',complete=False,passed=False,
            created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=digest(__file__),
            checks=[],hash_checks=[],numeric_comparisons=0,max_numeric_abs_error=0.,tensor_comparisons=0,
            observations_read=False,raw_target_reference_read=False,likelihood_recomputed=False,
            scientific_protocol_changed=False,failures=[])

    def need(self,condition,label):
        if not condition:raise AssertionError(label)

    def hashed(self,path,expected=None):
        actual=digest(path)
        self.need(expected is None or actual==expected,'SHA mismatch '+str(path))
        self.report['hash_checks'].append(dict(path=str(Path(path).resolve()),sha256=actual,bound=expected is not None))
        return actual

    def load(self,path,expected=None):
        actual=self.hashed(path,expected);value=torch.load(path,map_location='cpu',weights_only=False)
        self.need(digest(path)==actual,'Checkpoint changed during read');return value

    def number(self,a,b,label,atol=1e-10,rtol=1e-10):
        a,b=float(a),float(b);self.need(math.isfinite(a) and math.isfinite(b),'Nonfinite '+label)
        error=abs(a-b);self.report['numeric_comparisons']+=1
        self.report['max_numeric_abs_error']=max(error,self.report['max_numeric_abs_error'])
        self.need(error<=atol+rtol*abs(b),f'{label}: {a} != {b}')

    def array(self,a,b,label,atol=1e-10,rtol=1e-10):
        a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        self.need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'Shape/nonfinite '+label)
        error=np.abs(a-b);self.report['numeric_comparisons']+=a.size
        self.report['max_numeric_abs_error']=max(float(error.max(initial=0.)),self.report['max_numeric_abs_error'])
        self.need(bool(np.all(error<=atol+rtol*np.abs(b))),'Array differs '+label)

    def state(self,a,b,label):
        self.need(set(a)==set(b),'Tensor keys '+label)
        for key in a:
            self.need(torch.is_tensor(a[key]) and torch.is_tensor(b[key]),'Non-tensor '+label)
            self.need(a[key].shape==b[key].shape and torch.equal(a[key],b[key].to(a[key])),'Tensor differs '+label+'/'+key)
            self.report['tensor_comparisons']+=1

    def checkpoint_field(self,ck,run,budget,mode,item,protocol,prob):
        self.need(ck['arm']==run['arm'] and ck['seed']==run['seed'],'Field identity')
        self.number(ck['U'],run['U'],'Field U')
        self.need(ck['protocol_fingerprint']==protocol['fingerprint'],'Field protocol')
        self.need(ck['probability_sha256']==prob['sha256'],'Frozen probability source')
        self.need(ck['network_steps']==0,'Probability must not update')
        self.need(ck['step']==item['step'] and 0<=ck['step']<=budget,'Selection budget')
        if mode=='terminal':self.need(ck['step']==budget,'Exact budget terminal')
        state=ck['field_state']
        self.need(all(torch.is_tensor(v) and torch.isfinite(v).all() for v in state.values()),'Finite field tensors')
        self.need(not any('eta' in k or 'probability' in k or 'closure' in k for k in state),'No learned process/closure/probability inside field')
        buffers={'cold','scale','frequency','node_times','derivative_matrix'}
        expected={k:v.numel() for k,v in state.items() if k not in buffers}
        self.need(ck['field_trainable']==expected,'Only field trainable parameters')
        if run['arm']=='global_spline':self.need(set(expected)=={'controls'} and expected['controls']==96*2*624,'Spline global degrees of freedom')
        else:self.need('output.weight' in expected and any(k.startswith('hidden.') for k in expected),'PINN entire-field neural weights')
        for key,shape in [('a',(96,2,624)),('a_mid',(95,2,624)),('adot_mid',(95,2,624))]:
            self.need(tuple(ck[key].shape)==shape and bool(torch.isfinite(ck[key]).all()),'Saved trajectory '+key)

    def probability(self,C,protocol):
        gate_path=OUT/'probability_selection/selection_frozen.json'
        gate=read(gate_path);self.hashed(gate_path)
        plan_path=OUT/'probability_selection/selection_plan.json'
        self.hashed(plan_path,gate['selection_plan_sha256']);plan=read(plan_path)
        self.need(gate['complete'] and not gate['target_H_scored'] and not gate['target_reference_read'] and not gate['probability_parameters_updated'],'Non-target frozen probability selection')
        self.need(plan['case']=='vn0_0065m_s' and not plan['target_cases_allowed'],'Non-target selection case')
        self.need(plan['primary']['degree']==20,'Prospective selection quadrature')
        for key in ('parent_protocol','parent_training_gate','baseline_protocol','baseline_training_gate','selector_source'):
            self.hashed(plan[key]['path'],plan[key]['sha256'])
        for inp in plan['inputs']:self.hashed(inp['path'],inp['sha256'])
        scores=[]
        for item in gate['scores']:
            self.hashed(item['path'],item['sha256']);score=read(item['path'])
            self.need(score['selection_plan_sha256']==gate['selection_plan_sha256'],'Probability score plan')
            self.number(score['balanced'],.5*(score['free']+score['cap']),'Balanced probability NLL')
            scores.append(score)
        self.need(len(scores)==27,'Nine probability networks and three query/degree conditions')
        primary=[r for r in scores if r['label']=='anchor_gh20']
        means={f:float(np.mean([r['balanced'] for r in primary if r['family']==f])) for f in plan['primary']['tie_order']}
        best=min(means.values());winner=next(f for f in plan['primary']['tie_order'] if means[f]<=best+plan['primary']['tie_tolerance'])
        self.need(gate['family']==winner,'Probability family selection recomputed')
        models={r['field_seed']:r for r in gate['models']}
        self.need(set(models)==set(SEEDS),'Three frozen probability pairings')
        for seed,record in models.items():
            ck=self.load(record['path'],record['sha256'])
            self.need(ck['field_seed']==seed and ck['family']==winner and ck['selection_plan_sha256']==gate['selection_plan_sha256'],'Probability package identity')
            net,_=C.N.baseline_network(seed)
            net.load_state_dict(ck['net_state'],strict=True);net.freeze()
            self.need(not any(p.requires_grad or p.grad is not None for p in net.parameters()),'Frozen probability no gradient parameters')
            self.state(net.state_dict(),ck['net_state'],'Frozen probability reloaded')
            source=record['source']
            if winner!='baseline':
                raw=source['selected_model'];original=self.load(raw['path'],raw['sha256'])
                self.state(ck['net_state'],original['net_state'],'Original probability source')
            else:
                self.state(ck['net_state'],C.N.baseline_network(seed)[0].state_dict(),'Original baseline source')
        self.report['probability_selection']=dict(family=winner,primary_balanced_family_means=means,
            selection_sha256=digest(gate_path),wall_seconds=gate['wall_seconds'],cpu_seconds=gate['cpu_seconds'])
        self.report['checks'].append('Fixed non-target probability selection, original weights and frozen reload')
        return gate,models

    def fields(self,C,gate,protocol,models):
        expected={(c,a,s) for c in CASES for a in ARMS for s in SEEDS}
        self.need(gate['complete'] and gate['all_required_selections_frozen'] and gate['all_terminal_states_bound_before_evaluation'],'Global field gate flags')
        self.need(not gate['target_H_used_for_selection'] and not gate['target_reference_used_for_selection'],'No target scoring in field selection')
        self.need(gate['protocol_fingerprint']==protocol['fingerprint'],'Global field protocol')
        self.need(len(gate['runs'])==12 and {(r['case'],r['arm'],r['seed']) for r in gate['runs']}==expected,'12 frozen runs')
        self.hashed(OUT/'probability_selection/selection_frozen.json',gate['probability_gate_sha256'])
        index={};costs=[]
        for run in gate['runs']:
            self.hashed(run['status_path'],run['status_sha256']);status=read(run['status_path'])
            last=self.load(run['last_path'],run['last_sha256'])
            self.need(status['complete'] and status['step']==last['step']==600,'Complete 600-step run')
            self.need(status['network_steps']==last['network_steps']==0 and status['network_parameters_equal'],'Runtime probability freeze audit')
            self.need(status['probability_sha256']==last['probability_sha256']==models[run['seed']]['sha256'],'Runtime probability SHA')
            trace=last['trace'];self.need([r['step'] for r in trace]==list(range(601)),'Complete update trace 0..600')
            warm=last['warm'];self.need(trace[0]['validation']==warm,'Validation baseline')
            for budget in BUDGETS:
                best=trace[0]
                for row in trace[1:budget+1]:
                    if 'validation' not in row:continue
                    v=row['validation'];eligible=all(math.isfinite(v[k]) for k in ('total','free','cap')) and all(v[k]<=warm[k]+.01 for k in ('free','cap'))
                    self.need(row['eligible']==eligible,'Validation two-state guard')
                    if eligible and v['total']<best['validation']['total']-1e-6:best=row
                for mode in MODES:
                    item=run['budgets'][str(budget)][mode];ck=self.load(item['path'],item['sha256'])
                    self.checkpoint_field(ck,run,budget,mode,item,protocol,models[run['seed']])
                    expected_step=budget if mode=='terminal' else best['step']
                    self.need(ck['step']==expected_step,'Independent V-selected update')
                    for key in ('total','free','cap'):self.number(ck['val'][key],trace[expected_step]['validation'][key],'Saved validation')
                    index[(run['case'],run['arm'],run['seed'],budget,mode)]=(item,ck)
                terminal=index[(run['case'],run['arm'],run['seed'],budget,'terminal')][1]
                if budget==600:self.state(terminal['field_state'],last['field_state'],'Fixed terminal state')
            self.number(status['wall_seconds'],last['wall_seconds'],'Run wall accounting')
            self.number(status['cpu_seconds'],last['cpu_seconds'],'Run CPU accounting')
            costs.append(dict(case=run['case'],arm=run['arm'],seed=run['seed'],
                              wall_seconds=last['wall_seconds'],cpu_seconds=last['cpu_seconds'],
                              wall_at_200=trace[200]['wall_seconds'],cpu_at_200=trace[200]['cpu_seconds']))
        self.report['new_run_costs']=costs
        self.report['unique_new_inversion_worker_wall_seconds']=sum(r['wall_seconds'] for r in costs)
        self.report['unique_new_inversion_cpu_seconds']=sum(r['cpu_seconds'] for r in costs)
        self.report['cost_note']='200-step checkpoints are nested within the 12 600-step runs and are not counted twice; worker wall sum is not pipeline elapsed time'
        self.report['checks'].append('48 primary/terminal checkpoints, V-only selection, budgets, frozen probability, entire-field parameters')
        return index

    def observations(self,score,label):
        self.number(score['total'],.5*(score['free']+score['cap']),label+' balanced total',atol=1e-9)
        for kind in ('free','cap'):
            pids=score['per_pid'][kind]
            self.need(len(pids)>0 and all(int(pid)%5==0 for pid in pids),'H-only physical PIDs '+label)
            for key in ('nll','coverage','spread'):
                vals=[row[key] for row in pids.values()]
                actual=score[kind] if key=='nll' else score[key][kind]
                self.number(actual,np.mean(vals),label+'/'+kind+'/'+key,atol=1e-8)
            self.need(0<=score['coverage'][kind]<=1 and score['spread'][kind]>0,'Finite coverage/width '+label)

    def pde(self,pde,U):
        self.need(pde['physics']==dict(nu_s=0.,Fns=1.) and not pde['selection_or_optimization_use'],'Fixed deterministic PDE and diagnostic-only check')
        paths=pde['paths']
        self.need([r['interval_fraction'] for r in paths]==[0.,.25,.5,.75,1.],'Five fixed residual paths')
        self.need(all(r['points']==95 for r in paths) and pde['independent_points']==190 and pde['trained_unique_points']==191,'Independent collocation counts')
        for row in paths:
            self.array(row['dt_velocity_rms_mm_s'],np.asarray(row['dt_coefficient_rms'])*np.sqrt(624/2)*U*1000,'Residual Parseval velocity scale')
            self.array(row['instantaneous_acceleration_rms_m_s2'],np.asarray(row['dt_velocity_rms_mm_s']),'1ms increment mm/s and acceleration m/s2 scale')
            self.need(row['normal_B_energy']>=0 and row['super_std_energy']>=0,'Nonnegative residual energies')
        for name,indices,weights in [('trained_simpson',(0,2,4),(1/6,4/6,1/6)),('independent_quarter',(1,3),(.5,.5))]:
            result=pde[name];self.array(result['path_weights'],weights,'Residual integration path weights')
            for key in ('dt_coefficient_rms','dt_velocity_rms_mm_s'):
                expected=np.sqrt(sum(w*np.asarray(paths[i][key])**2 for i,w in zip(indices,weights)))
                self.array(result[key],expected,'Weighted residual RMS '+name+'/'+key)
            for key in ('normal_B_energy','super_std_energy'):
                self.number(result[key],sum(w*paths[i][key] for i,w in zip(indices,weights)),'Weighted residual energy '+name+'/'+key)
        for key in ('normal_B_energy','super_std_energy'):
            self.number(pde['independent_to_trained_energy_ratio'][key],pde['independent_quarter'][key]/max(pde['trained_simpson'][key],1e-30),'Independent residual ratio '+key)

    def summary(self,summary,index,C):
        rows=summary['runs'];expected=set(index)
        expected|={(case,'historical_joint',seed,200,mode) for case in CASES for seed in SEEDS for mode in MODES}
        self.need(len(rows)==60 and {(r['case'],r['arm'],r['seed'],int(r['budget']),r['mode']) for r in rows}==expected,'48 new plus 12 external rows')
        oldgate=read(C.NOUT/'selection_frozen.json');old_index={(r['case'],r['seed']):r for r in oldgate['runs'] if r['arm']=='joint'}
        gh=[];historical_costs=[]
        for r in rows:
            key=(r['case'],r['arm'],r['seed'],int(r['budget']),r['mode'])
            if r['arm'] in ARMS:
                item,ck=index[key]
                self.need(Path(r['checkpoint_path']).resolve()==Path(item['path']).resolve(),'Evaluation/checkpoint identity')
                self.need(r['checkpoint_sha256']==item['sha256'],'Evaluation/checkpoint SHA')
                self.need(int(r.get('iteration',r.get('step')))==ck['step'],'Evaluation step')
                self.need(r['source']=='new','New row provenance')
                self.pde(r['pde_collocation_audit'],CASES[r['case']])
                cost=next(c for c in self.report['new_run_costs'] if (c['case'],c['arm'],c['seed'])==key[:3])
                suffix='' if int(r['budget'])==600 else '_at_200'
                self.number(r['wall_seconds'],cost['wall_seconds'] if not suffix else cost['wall_at_200'],'New budget wall cost')
                self.number(r['cpu_seconds'],cost['cpu_seconds'] if not suffix else cost['cpu_at_200'],'New budget CPU cost')
            else:
                old=old_index[(r['case'],r['seed'])]
                path=old['path'] if r['mode']=='primary' else old['last_path']
                checksum=old['sha256'] if r['mode']=='primary' else old['last_sha256']
                self.need(Path(r['checkpoint_path']).resolve()==Path(path).resolve(),'Historical checkpoint identity')
                self.need(r['checkpoint_sha256']==checksum and r['source']=='historical','Historical checkpoint SHA/provenance')
                ck=self.load(path,checksum)
                self.need(int(r.get('iteration',r.get('step')))==ck['iteration'],'Historical selected/terminal step')
                prob=C.probability_record(r['seed']);probck=self.load(prob['path'],prob['sha256'])
                self.state(ck['net_state'],probck['net_state'],'Historical joint shares selected probability')
                self.need(r['pde_collocation_audit'] is None,'Historical external comparison has no new PDE diagnosis')
                if r['mode']=='terminal':historical_costs.append(dict(case=r['case'],seed=r['seed'],wall_seconds=ck['wall_seconds'],cpu_seconds=ck['cpu_seconds']))
            field=r['field'];curve=np.asarray(field['time_rms_mm_s'])
            self.need(curve.shape==(96,2),'96-frame field metric curve')
            self.array(field['rms_mm_s'],np.sqrt((curve**2).mean(0)),'Full-window RMS from time curve')
            self.array(field['initial_rms_mm_s'],curve[0],'Initial point RMS')
            self.array(field['relative_t5ms_rms_mm_s'],curve[5],'Relative 5ms point RMS')
            self.array(field['initial_5ms_window_rms_mm_s'],np.sqrt((curve[:6]**2).mean(0)),'First six-node RMS')
            main,higher=r['observations']['H'],r['heldout_degree28']
            self.observations(main,str(key)+'/20');self.observations(higher,str(key)+'/28')
            q=r['quadrature'];self.need(q['degree_primary']==20 and q['degree_comparison']==28,'Integration orders')
            for kind in ('total','free','cap'):self.number(q['nll_28_minus_20'][kind],higher[kind]-main[kind],'Quadrature NLL difference')
            for kind in ('free','cap'):
                self.number(q['coverage_28_minus_20'][kind],higher['coverage'][kind]-main['coverage'][kind],'Quadrature coverage difference')
                self.number(q['spread_um_28_minus_20'][kind],higher['spread'][kind]-main['spread'][kind],'Quadrature width difference')
            gh.append(abs(higher['cap']-main['cap']))
        self.report['max_abs_cap_nll_GH28_minus_GH20']=max(gh)
        self.report['historical_run_costs']=historical_costs
        self.report['checks'].append('60 result rows, per-PID probability arithmetic, field time reductions and quadrature sensitivity')
        return rows

    def flat(self,row,degree=20):
        h=row['observations']['H'] if degree==20 else row['heldout_degree28'];f=row['field']
        values=dict(normal_rms_mm_s=f['rms_mm_s'][0],superfluid_rms_mm_s=f['rms_mm_s'][1],heldout_total_nll=h['total'])
        for i,fluid in enumerate(('normal','superfluid')):
            for prefix,key in [('initial','initial_rms_mm_s'),('relative_t5ms','relative_t5ms_rms_mm_s'),('initial_5ms_window','initial_5ms_window_rms_mm_s')]:
                values[prefix+'_'+fluid+'_rms_mm_s']=f[key][i]
        for kind in ('free','cap'):
            values.update({f'heldout_{kind}_nll':h[kind],f'heldout_{kind}_coverage':h['coverage'][kind],
                           f'heldout_{kind}_coverage_abs_error':abs(h['coverage'][kind]-.9),f'heldout_{kind}_spread_um':h['spread'][kind]})
        return values

    def described(self,actual,values,label):
        self.number(actual['mean'],np.mean(values),label+' mean')
        self.number(actual['min'],min(values),label+' min');self.number(actual['max'],max(values),label+' max')
        self.need(set(actual['by_seed'])==set(map(str,SEEDS)),'Exact seed mapping '+label)
        for seed,value in zip(SEEDS,values):self.number(actual['by_seed'][str(seed)],value,label+' by_seed')

    def reductions(self,result,rows):
        index={(r['case'],r['arm'],r['seed'],int(r['budget']),r['mode']):r for r in rows}
        pairs=(('pinn','global_spline'),('pinn','historical_joint'),('global_spline','historical_joint'))
        core=('normal_rms_mm_s','superfluid_rms_mm_s','heldout_free_nll','heldout_cap_nll')
        for budget in BUDGETS:
            for mode in MODES:
                for degree in (20,28):
                    for case in CASES:
                        section=result['summary'][str(budget)][mode][str(degree)][case]
                        vals={arm:[self.flat(index[(case,arm,seed,200 if arm=='historical_joint' else budget,mode)],degree) for seed in SEEDS]
                              for arm in (*ARMS,'historical_joint')}
                        for arm,values in vals.items():
                            self.need(set(section['arms'][arm])==set(values[0]),'Complete summary metrics')
                            for key in values[0]:self.described(section['arms'][arm][key],[v[key] for v in values],'Arm summary '+key)
                        for a,b in pairs:
                            comparison=section['paired_changes'][a+'_minus_'+b]
                            for key in vals[a][0]:self.described(comparison['metrics'][key],[x[key]-y[key] for x,y in zip(vals[a],vals[b])],'Paired change '+key)
                            count=sum(all(x[k]<y[k] for k in core) for x,y in zip(vals[a],vals[b]))
                            self.need(comparison['joint_four_metric_improvement_count']==count,'Joint four-metric seed count')
                            self.need(comparison['all_four_mean_deltas_negative']==all(comparison['metrics'][k]['mean']<0 for k in core),'Joint four-metric mean conclusion')
        for row in result['quadrature_directions']:
            values=[result['summary'][str(row['budget'])][row['mode']][str(d)][row['case']]['paired_changes'][row['pair']]['metrics'][row['metric']]['mean'] for d in (20,28)]
            self.number(row['delta20'],values[0],'GH20 paired mean');self.number(row['delta28'],values[1],'GH28 paired mean')
            self.need(row['sign_reversal']==(values[0]*values[1]<0),'Mean integration-direction reversal')
        self.need(len(result['terminal_minus_primary'])==len(result['budget600_minus200'])==8,'Eight paired terminal/budget reductions')
        for row in result['terminal_minus_primary']:
            left=[self.flat(index[(row['case'],row['arm'],s,row['budget'],'terminal')]) for s in SEEDS]
            right=[self.flat(index[(row['case'],row['arm'],s,row['budget'],'primary')]) for s in SEEDS]
            for key in left[0]:self.described(row['metrics'][key],[a[key]-b[key] for a,b in zip(left,right)],'Terminal-primary '+key)
        for row in result['budget600_minus200']:
            left=[self.flat(index[(row['case'],row['arm'],s,600,row['mode'])]) for s in SEEDS]
            right=[self.flat(index[(row['case'],row['arm'],s,200,row['mode'])]) for s in SEEDS]
            for key in left[0]:self.described(row['metrics'][key],[a[key]-b[key] for a,b in zip(left,right)],'600-200 '+key)
        self.report['checks'].append('Independent all-arm summaries, paired changes, joint-improvement counts and 200/600 comparisons')

    def auxiliary(self,auxiliary,rows):
        self.need(auxiliary['complete'] and auxiliary['probability_unchanged'] and not auxiliary['fitting_or_selection'],'Frozen-query attribution flags')
        self.need(len(auxiliary['rows'])==24,'24 fixed-old-node query/field attributions')
        index={(r['case'],r['arm'],r['seed'],int(r['budget']),r['mode']):r for r in rows}
        expected={(c,a,s,b) for c in CASES for a in ARMS for s in SEEDS for b in BUDGETS}
        self.need({(r['case'],r['arm'],r['seed'],r['budget']) for r in auxiliary['rows']}==expected,'Attribution pair coverage')
        def flat_score(h):
            return dict(**{k+'_nll':h[k] for k in ('total','free','cap')},**{f'{s}_{k if k=="coverage" else "spread_um"}':h[k][s] for s in ('free','cap') for k in ('coverage','spread')})
        for item in auxiliary['rows']:
            old=index[(item['case'],'historical_joint',item['seed'],200,'primary')]
            new=index[(item['case'],item['arm'],item['seed'],item['budget'],'primary')]
            self.need(item['fixed_historical_checkpoint_sha256']==old['checkpoint_sha256'] and item['own_checkpoint_sha256']==new['checkpoint_sha256'],'Attribution field SHA')
            for degree in ('20','28'):
                part=item['by_degree'][degree]
                b,x,n=[flat_score(part[k]) for k in ('historical_original_query','historical_fixed_nodes_continuous_query','new_field_continuous_query')]
                hb=old['observations']['H'] if degree=='20' else old['heldout_degree28']
                hn=new['observations']['H'] if degree=='20' else new['heldout_degree28']
                for key,value in flat_score(hb).items():self.number(b[key],value,'Attribution original endpoint')
                for key,value in flat_score(hn).items():self.number(n[key],value,'Attribution new endpoint')
                self.observations(part['historical_fixed_nodes_continuous_query'],'Fixed historical continuous query')
                for key in b:
                    c=part['components']
                    self.number(c['query_definition_change_at_fixed_nodes'][key],x[key]-b[key],'Query change')
                    self.number(c['field_change_under_continuous_query'][key],n[key]-x[key],'Field change at new query')
                    self.number(c['total_change'][key],n[key]-b[key],'Total query plus field change')
                    self.number(c['query_definition_change_at_fixed_nodes'][key]+c['field_change_under_continuous_query'][key],c['total_change'][key],'Attribution additive identity')
        self.report['checks'].append('24 fixed-old-node query-definition and new-field descriptive attributions')

    def exports(self,manifest,rows):
        artifacts=manifest['artifacts'];self.need(len(artifacts)==22,'20 reconstructed plus 2 reference NPZ exports')
        refs={};fields=[]
        for item in artifacts:
            self.hashed(item['path'],item['sha256']);self.need(item['frames']==96,'All exports have 96 frames')
            z=np.load(item['path'],allow_pickle=False)
            self.need(z['a'].shape==(96,2,624) and z['velocity_m_s'].shape==(96,2,40,40,2),'Export coefficient/grid shape')
            self.array(z['relative_time_s'],np.arange(96)*.001,'Relative export time')
            self.array(z['time_s'],.005+np.arange(96)*.001,'Physical export time')
            self.need(all(np.isfinite(z[k]).all() for k in z.files if np.issubdtype(z[k].dtype,np.number)),'Finite NPZ numeric arrays')
            values={key:z[key] for key in z.files};z.close()
            if item['kind']=='reference_npz':refs[item['case']]=values
            else:fields.append((item,values))
        self.need(set(refs)==set(CASES) and len(fields)==20,'Export case coverage')
        for item,z in fields:
            ref=refs[item['case']]
            self.array(z['velocity_error_m_s'],z['velocity_m_s']-ref['velocity_m_s'],'Export velocity difference',atol=1e-11)
            row=next(r for r in rows if (r['case'],r['arm'],r['seed'],int(r['budget']),r['mode'])==
                     (item['case'],item['arm'],301030,int(item['budget']),item['mode']))
            err=(z['a']-ref['a'])*CASES[item['case']]*1000
            curve=np.sqrt(.5*(err**2).sum(-1))
            self.array(curve,row['field']['time_rms_mm_s'],'Export Parseval RMS')
            grid=np.sqrt((z['velocity_error_m_s']**2).sum(-1).mean((-1,-2)))*1000
            maxdiff=float(np.max(np.abs(grid-curve)))
            full_grid=np.sqrt((grid**2).mean(0))
            full_diff=float(np.max(np.abs(full_grid-np.asarray(row['field']['rms_mm_s']))))
            self.number(item['grid_vs_summary_rms_max_mm_s'],full_diff,'Grid/summary recorded difference',atol=1e-9)
            self.need(maxdiff<1e-7,'Matched grid/modal RMS identity')
        self.report['checks'].append('22 complete 96-frame exports, coefficient and velocity RMS identities')

    def plots(self,manifest):
        self.need(manifest['complete'] and manifest['figure_groups']==24,'24 completed figure groups')
        self.hashed(OUT/'PINN_EVALUATION_SUMMARY.json',manifest['evaluation_summary_sha256'])
        self.hashed(OUT/'FIELD_MANIFEST.json',manifest['field_manifest_sha256'])
        self.hashed(OUT/'selection_frozen.json',manifest['selection_gate_sha256'])
        items=manifest['artifacts'];self.need(len(items)==48,'24 PNG/SVG pairs')
        self.need(len({r['name'] for r in items})==24,'Unique figure groups')
        for item in items:
            self.hashed(item['path'],item['sha256']);self.need(Path(item['path']).stat().st_size>1000,'Nonempty figure')
        for name in {r['name'] for r in items}:self.need({r['format'] for r in items if r['name']==name}=={'png','svg'},'PNG/SVG pair')
        self.report['checks'].append('24 figure groups, exact evaluation/export provenance and PNG/SVG hashes')

    def run(self):
        # Check field-gate completeness before importing any inherited modules.
        gate=read(OUT/'selection_frozen.json')
        self.need(gate.get('complete') and len(gate.get('runs',[]))==12,'Require global freeze before audit')
        from pinn_common import verify_protocol
        import pinn_common as C
        protocol=verify_protocol();self.hashed(OUT/'protocol.json');self.hashed(OUT/'selection_frozen.json')
        self.report['protocol_fingerprint']=protocol['fingerprint']
        probability,models=self.probability(C,protocol)
        index=self.fields(C,gate,protocol,models)
        summary=read(OUT/'PINN_EVALUATION_SUMMARY.json');self.hashed(OUT/'PINN_EVALUATION_SUMMARY.json')
        rows=self.summary(summary,index,C)
        self.reductions(summary,rows)
        self.auxiliary(summary['auxiliary'],rows)
        self.exports(read(OUT/'FIELD_MANIFEST.json'),rows)
        self.plots(read(OUT/'PLOT_MANIFEST.json'))
        self.report['complete']=True;self.report['passed']=True
        return self.report


def selftest():
    """Manufactured arithmetic and deliberate corruption; opens no data files."""
    spec=importlib.util.spec_from_file_location('pinn_evaluation_audit_selftest',HERE/'evaluate_pinn.py')
    evaluation=importlib.util.module_from_spec(spec);sys.modules[spec.name]=evaluation;spec.loader.exec_module(evaluation)
    audit=Audit();rows=[]
    for ci,(case,U) in enumerate(CASES.items()):
        for ai,arm in enumerate((*ARMS,'historical_joint')):
            for si,seed in enumerate(SEEDS):
                for budget in ((200,) if arm=='historical_joint' else BUDGETS):
                    for mi,mode in enumerate(MODES):
                        shift=.01*(ci+ai+si+mi)+budget*1e-5
                        curve=np.ones((96,2))*np.array([1+shift,2+shift])
                        field=dict(rms_mm_s=curve[0].tolist(),initial_rms_mm_s=curve[0].tolist(),
                                   relative_t5ms_rms_mm_s=curve[5].tolist(),time_rms_mm_s=curve.tolist(),
                                   initial_5ms_window_rms_mm_s=curve[0].tolist())
                        def h(extra):
                            free,cap=1+shift+extra,2+shift+extra
                            return dict(total=.5*(free+cap),free=free,cap=cap,coverage=dict(free=.8,cap=.95),spread=dict(free=2.,cap=3.),
                                        per_pid={kind:{'0':dict(nll=value,coverage=cov,spread=spread)} for kind,value,cov,spread in [('free',free,.8,2.),('cap',cap,.95,3.)]})
                        rows.append(dict(case=case,U=U,arm=arm,seed=seed,budget=budget,mode=mode,field=field,
                                         observations=dict(H=h(0)),heldout_degree28=h(.001)))
    summary,directions,terminal,budgets=evaluation.summarize(rows)
    result=dict(summary=summary,quadrature_directions=directions,terminal_minus_primary=terminal,budget600_minus200=budgets)
    audit.reductions(result,rows)
    audit.observations(rows[0]['observations']['H'],'Manufactured valid H')
    failures_detected=0
    corrupted=copy.deepcopy(rows[0]['observations']['H']);corrupted['cap']+=.1
    try:audit.observations(corrupted,'Deliberately corrupted H')
    except AssertionError:failures_detected+=1
    corrupted=copy.deepcopy(result);corrupted['summary']['200']['primary']['20']['vn0_0015m_s']['arms']['pinn']['normal_rms_mm_s']['mean']+=.1
    try:audit.reductions(corrupted,rows)
    except AssertionError:failures_detected+=1
    from types import SimpleNamespace
    class ManufacturedField:
        def coefficients_and_dt(self,t):
            return torch.sin(t[:,None,None]*20).expand(-1,2,624),20*torch.cos(t[:,None,None]*20).expand(-1,2,624)
    old=SimpleNamespace(theta=torch.zeros(2,624,dtype=torch.float64),op=SimpleNamespace(nu_s=0.,friction_factor=1.,U=.004),
                        prior=SimpleNamespace(whiten=lambda x:3*x),std=torch.ones(2,624,dtype=torch.float64))
    C=SimpleNamespace(CURRENT=SimpleNamespace(batched_rhs=lambda op,a:2*a))
    pde=evaluation.pde_collocation_audit(C,ManufacturedField(),old);audit.pde(pde,.004)
    corrupted=copy.deepcopy(pde);corrupted['independent_quarter']['normal_B_energy']*=1.1
    try:audit.pde(corrupted,.004)
    except AssertionError:failures_detected+=1
    if failures_detected!=3:raise AssertionError('Manufactured corruption escaped independent audit')
    return dict(passed=True,manufactured_rows=60,numeric_checks=audit.report['numeric_comparisons'],
                deliberate_corruptions_detected=failures_detected,no_target_or_reference_files_opened=True)


def main():
    audit=Audit()
    try:result=audit.run()
    except Exception as error:
        audit.report['failures'].append(type(error).__name__+': '+str(error))
        dump(OUT/'FINAL_INDEPENDENT_AUDIT.json',audit.report)
        raise
    dump(OUT/'FINAL_INDEPENDENT_AUDIT.json',result)
    lines=['PINN 二流场第一轮独立结果审计：PASS',
           '48行新场主/末态 + 12行历史外部参照；两个预算共享同一600步轨迹，不重复计成本。',
           '9网络非目标选家族、3冻结概率权重、12场完整反演与源/检查点SHA通过。',
           f"数值核查 {result['numeric_comparisons']} 项，最大绝对差 {result['max_numeric_abs_error']:.9g}。",
           f"捕获NLL的最大|GH28-GH20|={result['max_abs_cap_nll_GH28_minus_GH20']:.9g}；描述性敏感性不等于积分收敛。",
           f"新反演CPU总计 {result['unique_new_inversion_cpu_seconds']:.3f} 秒；各worker wall总计 {result['unique_new_inversion_worker_wall_seconds']:.3f} 秒。",
           '逐项检查：',*['- '+s for s in result['checks']],
           '未读取原始目标reference.pt或粒子观测，未重新计算似然，未进行选择或训练。',
           '本审计只检验完整性、算术、来源和冻结；物理效果与可辨识性需结合正式结果解释。']
    (OUT/'FINAL_INDEPENDENT_AUDIT.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('AUDIT_PASSED',result['numeric_comparisons'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--selftest',action='store_true');args=parser.parse_args()
    if args.selftest:print(json.dumps(selftest(),ensure_ascii=False,indent=2))
    else:main()
