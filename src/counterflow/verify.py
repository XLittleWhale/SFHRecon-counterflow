"""Bounded numerical equivalence and data-integrity checks; no optimization."""
import gc
import sys
from .common import *
from .paths import ROOT

def main():
    declare();net,cal=model();rows=[]
    for label in CASES:
        c,windows=bundle(label);op=rt.make(rt.load(c['operator_source']),'cpu')
        root=HERE/'fits'/label
        choice=readjson(root/'selected_joint.json');path=root/choice['chosen']
        theta=rt.load(path/'selected.pt')
        with torch.no_grad():
            a,q=rt.forward(c,op,theta);vel=rt.velocity(c,a)
            with np.load(path/'fields.npz') as old:
                ae=float(np.max(abs(a.numpy()-old['coefficients'])))
                ve=float(np.max(abs(vel.numpy()-old['velocity'])))
            assert ae<1e-9 and ve<1e-11,(label,ae,ve)
            score,_,values=validation(c,a,parts(c,1),net,cal)
            expected=rt.load(path/'validation_scores.pt')
            err=max(float((values[k]-expected[k]).abs().max()) for k in values)
            assert err<1e-7,(label,err)
        ids={g:{int(pid)%c['pid_stride'] for _,z in parts(c,g) for pid in z['pid']} for g in (0,1,2)}
        assert not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
        rows.append(dict(case=label,coefficient_max_abs=ae,velocity_max_abs_m_s=ve,
                         validation_score_max_abs=err,particle_split_disjoint=True))
        print('Verified',label,'field error',f'{ve:.2e}','score error',f'{err:.2e}',flush=True)
        del c,windows,op,a,q,theta,vel;gc.collect()
    # Validate the recorded hashes of the portable input bundle.
    manifest=readjson(CONFIG/'data_manifest.json')
    for relative,record in manifest.items():
        p=DATA/relative
        assert p.stat().st_size==record['bytes'] and rt.sha(p)==record['sha256'],relative
    imported=[str(m.__file__) for k,m in sys.modules.items() if k.startswith('counterflow') and getattr(m,'__file__',None)]
    assert all(Path(p).resolve().is_relative_to(ROOT/'src') for p in imported)
    result=dict(cases=rows,data_files_checked=len(manifest),package_modules=len(imported),
                historical_imports=False,long_training_executed=False,**fingerprints())
    rt.dump(HERE/'verification.json',result)
    print('All five cases and data hashes verified.',flush=True)
