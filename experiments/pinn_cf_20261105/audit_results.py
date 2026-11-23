"""Post-gate integrity and independent arithmetic audit. No fitting/selection."""
from cf_common import *
import math
import numpy as np
from cf_experiment import selected_from_trace

def close(a,b,atol=2e-9):
    if not math.isclose(float(a),float(b),rel_tol=2e-10,abs_tol=atol):
        raise AssertionError(f'Numerical accounting mismatch: {a} != {b}')

def audit():
    p,gate = evaluation_gate()
    summary = read(OUT/'CF_EVALUATION_SUMMARY.json')
    fields = read(OUT/'FIELD_MANIFEST.json')
    plots = read(OUT/'PLOT_MANIFEST.json')
    assert summary['complete'] and len(summary['runs'])==30
    assert len(fields['artifacts'])==10 and len(plots['artifacts'])==48 and plots['figure_groups']==24
    assert fields['evaluation_sha256']==plots['evaluation_sha256']==sha(OUT/'CF_EVALUATION_SUMMARY.json')
    assert plots['field_manifest_sha256']==sha(OUT/'FIELD_MANIFEST.json')
    checks = 0
    for run in gate['runs']:
        assert sha(run['status_path'])==run['status_sha256']
        assert sha(run['last_path'])==run['last_sha256']
        last = load(run['last_path'])
        assert last['step']==600 and last['network_steps']==0
        assert len(last['trace'])==601
        for budget in BUDGETS:
            picked = selected_from_trace(last['trace'],last['warm'],budget,p)
            assert run['budgets'][str(budget)]['primary']['step']==picked['step']
            assert run['budgets'][str(budget)]['terminal']['step']==budget
        for record in [run['background'],*[v for b in run['budgets'].values() for v in b.values()]]:
            ck = load(record['path'])
            assert ck['a'].shape==(101,2,325) and ck['a_half'].shape==(201,2,325)
            assert torch.isfinite(ck['a']).all() and torch.isfinite(ck['adot_half']).all()
            assert ck['network_steps']==0 and ck['probability_sha256']==run['probability_sha256']
            checks += 1
    expected={(case,seed,budget,mode) for case in CASES for seed in SEEDS
              for budget,mode in [(0,'background')]+[(b,m) for b in BUDGETS for m in ('primary','terminal')]}
    assert {(r['case'],r['seed'],r['budget'],r['mode']) for r in summary['runs']}==expected
    for row in summary['runs']:
        assert sha(row['checkpoint_path'])==row['checkpoint_sha256']
        assert row['network_steps']==0 and not row['reference_read'] and not row['target_test_used_for_selection']
        for label in ('heldout','heldout_degree28'):
            h = row[label]
            close(h['total'],.5*(h['free']+h['cap']))
            close(h['native_nll']['total'],.5*(h['native_nll']['free']+h['native_nll']['cap']))
            for kind in ('free','cap'):
                values = list(h['per_pid'][kind].values())
                assert values and all(r['count']>0 for r in values)
                close(h[kind],np.mean([r['nll'] for r in values]))
                close(h['native_nll'][kind],np.mean([r['native_nll'] for r in values]))
                close(h['coverage'][kind],np.mean([r['coverage'] for r in values]))
                close(h['spread'][kind],np.mean([r['spread'] for r in values]))
                close(h['rmse_um'][kind],np.sqrt(np.mean([r['squared_error'] for r in values])))
                assert 0<=h['coverage'][kind]<=1 and h['spread'][kind]>=0 and h['rmse_um'][kind]>=0
                checks += 7
        for kind in ('total','free','cap'):
            close(row['quadrature']['nll_per_ms'][kind],row['heldout_degree28'][kind]-row['heldout'][kind])
        close(row['pde']['exposure_ms'],p['scopes'][row['case']]['exposure_ms'])
        assert row['pde']['trained_unique_points']==201 and row['pde']['independent_points']==200
        boundary=row['field']['boundary']
        for key in ('normal_no_slip_max_mm_s','super_no_penetration_max_mm_s','axial_periodic_velocity_max_mm_s',
                    'axial_periodic_gradient_max_s_inv','divergence_max_s_inv'):
            assert boundary[key]<1e-7,(key,boundary[key])
        assert row['field']['field_truth_available'] is False
    for case,scenarios in summary['summary'].items():
        for label,metrics in scenarios.items():
            budget,mode=label.split('_',1)
            rr=[r for r in summary['runs'] if (r['case'],r['budget'],r['mode'])==(case,int(budget),mode)]
            assert len(rr)==3
            for kind in ('free','cap'):
                for suffix,path in [('nll_per_ms',[kind]),('native_nll',['native_nll',kind]),
                                    ('coverage90',['coverage',kind]),('spread_um',['spread',kind]),
                                    ('endpoint_rmse_um',['rmse_um',kind])]:
                    values=[]
                    for r in rr:
                        v=r['heldout']
                        for key in path: v=v[key]
                        values.append(v)
                    metric=metrics[f'H_{kind}_{suffix}']
                    close(metric['mean'],np.mean(values))
                    close(metric['sample_std'],np.std(values,ddof=1))
                    close(metric['minimum'],min(values));close(metric['maximum'],max(values))
                    checks+=4
    for item in fields['artifacts']:
        assert sha(item['path'])==item['sha256'] and item['frames']==101
        with np.load(item['path'],allow_pickle=False) as d:
            assert d['velocity_m_s'].shape==(101,2,32,64,2)
            assert d['vorticity_s_inv'].shape==(101,2,32,64)
            assert np.allclose(d['x_edges_m'][[0,-1]],[0,.016]) and np.allclose(d['y_edges_m'][[0,-1]],[0,.008])
            for key in d.files: assert np.isfinite(d[key]).all(),(item['path'],key)
    from PIL import Image
    dimensions=[]
    for item in plots['artifacts']:
        assert sha(item['path'])==item['sha256']
        if item['format']=='png':
            with Image.open(item['path']) as im:
                dimensions.append(dict(path=item['path'],pixels=list(im.size)))
                im.verify()
    max_gh = {k:max(abs(r['quadrature']['nll_per_ms'][k]) for r in summary['runs']) for k in ('free','cap')}
    result=dict(complete=True,passed=True,created_utc=utc(),protocol_fingerprint=p['fingerprint'],
        checkpoint_records=30,evaluation_rows=30,field_npz=10,frames_per_npz=101,figure_groups=24,
        numerical_accounting_checks=checks,probability_parameter_updates=0,
        maximum_GH20_28_nll_per_ms_difference=max_gh,plot_dimensions=dimensions,
        scientific_improvement_not_assumed=True,reference_field_available=False,
        human_style_independent_review_and_representative_visual_inspection_still_required=True)
    dump(OUT/'AUDIT_RESULTS.json',result)
    return result

if __name__=='__main__':
    setup()
    print(json.dumps(audit(),ensure_ascii=False,indent=2))
