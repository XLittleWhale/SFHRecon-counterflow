"""Numeric completion and matched field comparisons survive optional plot errors."""
import copy
import json
from pathlib import Path

import numpy as np

import evaluate as E


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')


def comparison_fixture(tmp_path):
    case='cf_2K_Q113'
    inputs={f'prepared/{case}/{name}':name+'_hash' for name in
            ('data_audit.json','inversion_observations.pt','heldout_observations.pt','reference_evaluation.npz')}
    protocol=dict(budgets=[150],fingerprint='baseline',inputs=inputs,field={})
    row=dict(case=case,seed=301030,label='150_primary',step=140,H_NLL_ms=999.,
             normal_rmse_m_s=.002,superfluid_rmse_m_s=.004,
             normal_relative_L2=.2,superfluid_relative_L2=.4)
    write(tmp_path/'protocol.json',protocol)
    write(tmp_path/'evaluation/summary.json',dict(complete=True,selected_budget=150,
                                               protocol_fingerprint='baseline',runs=[row]))
    write(tmp_path/'analysis/result_summary.json',dict(protocol_fingerprint='baseline'))
    current=copy.deepcopy(protocol);current['fingerprint']='current'
    current_row=dict(row,step=150,normal_rmse_m_s=.001,superfluid_rmse_m_s=.005,H_NLL_ms=-999.)
    return current,[current_row]


def test_matched_comparison_excludes_cross_model_nll(tmp_path):
    protocol,rows=comparison_fixture(tmp_path)
    result=E.comparison_150(rows,protocol,tmp_path)
    assert result['complete'] and not result['cross_model_NLL_comparison']
    row=result['rows'][0]
    assert row['normal']['rmse_change_percent']==-50.
    assert row['superfluid']['rmse_change_percent']==25.
    assert not any('NLL' in key or 'nll' in key for key in row)
    assert row['selected_step']==150 and row['baseline_selected_step']==140


def test_changed_inputs_or_budget_are_not_called_matched(tmp_path):
    protocol,rows=comparison_fixture(tmp_path)
    key=next(iter(protocol['inputs']))
    protocol['inputs'][key]='changed'
    result=E.comparison_150(rows,protocol,tmp_path)
    assert not result['complete'] and result['unmatched'] and not result['rows']
    protocol['budgets']=[200]
    result=E.comparison_150(rows,protocol,tmp_path)
    assert not result['complete'] and not result['rows']


def test_plot_import_failure_does_not_abort_numeric_evaluation(tmp_path,monkeypatch):
    case='cf_2K_Q113';seed=301030
    prepared=tmp_path/'prepared';prepared.mkdir()
    reference=dict(time_s=np.array([0.]),x_m=np.array([0.,.016]),y_m=np.array([0.,.004]),
                   vn=np.zeros((1,2,2,2)),vs=np.zeros((1,2,2,2)))
    np.savez(prepared/'reference_evaluation.npz',**reference)
    protocol=dict(fingerprint='test',field=dict(validation_degree=20),budgets=[150])
    record=dict(path='fake.pt',step=0,sha256='sha')
    gate=dict(runs=[dict(case=case,seed=seed,background=record,
                        budgets={'150':{'primary':record,'terminal':record}})])
    monkeypatch.setattr(E,'OUT',tmp_path)
    monkeypatch.setattr(E,'CASES',(case,))
    monkeypatch.setattr(E,'BUDGETS',(150,))
    monkeypatch.setattr(E,'evaluate_gate',lambda:(protocol,gate))
    monkeypatch.setattr(E,'prepared',lambda _:prepared)
    def load(path):
        if Path(path).name=='inversion_observations.pt':
            return {'meta':{'physics_config':{},'intervals':1}}
        if Path(path).name=='heldout_observations.pt':
            return {'z':{}}
        return {'step':0,'field_state':{}}
    monkeypatch.setattr(E,'load',load)
    monkeypatch.setattr(E,'device',lambda:'cpu')
    monkeypatch.setattr(E,'move_tensors',lambda z:z)
    monkeypatch.setattr(E,'probability',lambda *a,**kw:(None,None,None,'digest'))
    monkeypatch.setattr(E.F,'prepare_physics',lambda *a,**kw:({},None,None,None))
    class Field:
        def load_state_dict(self,state):pass
    monkeypatch.setattr(E.F,'make_field',lambda *a,**kw:Field())
    monkeypatch.setattr(E.R,'score',lambda *a,**kw:dict(total=1.,free=1.,cap=1.))
    errors={name:dict(vector_rmse_m_s=.001,relative_L2=.2) for name in ('normal','superfluid')}
    monkeypatch.setattr(E,'field_metrics',lambda *a,**kw:(errors,np.zeros((1,2,2,2,2))))
    def fail(*args,**kw):raise ImportError('simulated matplotlib dependency failure')
    monkeypatch.setattr(E,'plot',fail)
    E.main()
    summary=json.loads((tmp_path/'evaluation/summary.json').read_text())
    assert summary['complete'] and summary['numeric_evaluation_complete']
    assert not summary['plots_complete'] and len(summary['runs'])==3
    assert (tmp_path/'evaluation'/case/f'seed_{seed}/150_terminal_fields.npz').is_file()
    status=json.loads((tmp_path/'evaluation'/case/f'seed_{seed}/plot_status.json').read_text())
    assert status['pending'] and 'ImportError' in status['error']


def test_plots_only_uses_exports_without_model_loading(tmp_path,monkeypatch):
    case='cf_2K_Q113';seed=301030
    prepared=tmp_path/'prepared';prepared.mkdir()
    folder=tmp_path/'evaluation'/case/f'seed_{seed}';folder.mkdir(parents=True)
    grid=dict(time_s=np.array([0.]),x_m=np.array([0.,.016]),y_m=np.array([0.,.004]))
    data=dict(**grid,vn=np.zeros((1,2,2,2)),vs=np.zeros((1,2,2,2)))
    np.savez(prepared/'reference_evaluation.npz',**data)
    np.savez(folder/'150_primary_fields.npz',**data)
    write(folder/'150_primary.json',{'field':{}})
    write(tmp_path/'evaluation/summary.json',dict(complete=True,protocol_fingerprint='test',
        selected_budget=150,runs=[dict(case=case,seed=seed,label='150_primary')]))
    monkeypatch.setattr(E,'OUT',tmp_path)
    monkeypatch.setattr(E,'prepared',lambda _:prepared)
    monkeypatch.setattr(E,'verify_protocol',lambda:dict(fingerprint='test'))
    def forbidden(*args,**kwargs):raise AssertionError('Plot recovery must not load or score a model')
    monkeypatch.setattr(E,'probability',forbidden)
    monkeypatch.setattr(E.F,'prepare_physics',forbidden)
    monkeypatch.setattr(E.R,'score',forbidden)
    seen=[]
    monkeypatch.setattr(E,'plot',lambda *args:seen.append(args[3].shape))
    status=E.plots_only()
    assert status[0]['complete'] and seen==[(1,2,2,2,2)]
