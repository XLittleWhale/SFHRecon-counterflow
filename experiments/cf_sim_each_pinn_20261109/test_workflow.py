"""Selection and lifecycle checks that protect final held-out evaluation."""
import json
import os
import psutil
import pytest
import reconstruct as R
import run_pipeline as P

SETTINGS={'field':{'guard_nll_per_ms':.01,'selection_delta_nll_per_ms':1e-6}}

def test_state_guard_blocks_balanced_improvement_that_harms_one_state():
    warm={'total':2.,'free':1.,'cap':3.}
    assert not R.eligible({'total':1.8,'free':1.011,'cap':2.589},warm,SETTINGS)
    assert R.eligible({'total':1.8,'free':1.009,'cap':2.591},warm,SETTINGS)

def test_budget_selection_excludes_future_steps_and_ineligible_rows():
    warm={'total':2.,'free':1.,'cap':3.}
    trace=[dict(step=0,validation=warm),
           dict(step=10,validation={'total':1.9,'free':.9,'cap':2.9}),
           dict(step=20,validation={'total':1.,'free':1.02,'cap':.98}),
           dict(step=210,validation={'total':.8,'free':.6,'cap':1.})]
    assert R.selected_from_trace(trace,warm,200,SETTINGS)['step']==10
    assert R.selected_from_trace(trace,warm,600,SETTINGS)['step']==210

def test_live_lock_refused_and_dead_lock_archived(tmp_path):
    path=tmp_path/'job.lock'
    path.write_text(json.dumps({'pid':os.getpid(),'process_created':psutil.Process().create_time()}))
    with pytest.raises(RuntimeError): P.acquire_lock(path)
    path.write_text(json.dumps({'pid':os.getpid(),'process_created':0.}))
    P.acquire_lock(path)
    assert json.loads(path.read_text())['pid']==os.getpid()
    assert len(list(tmp_path.glob('job.lock.stale.*')))==1
