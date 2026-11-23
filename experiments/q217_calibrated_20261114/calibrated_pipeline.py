"""217 frozen probability -> calibrated PINN -> V freeze -> H/field evaluation."""
import argparse
import os
import json
from pathlib import Path
import torch
from runtime import read,dump,sha,lock
import calibrated_workflow as workflow
import calibrated_fields as fields


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','validate','run','fields','evaluate','smoke'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--max-new-steps',type=int)
    args=parser.parse_args();out=args.output.resolve()
    if args.max_new_steps is not None and args.max_new_steps<1:raise ValueError('Positive max-new-steps required')
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    os.environ.setdefault('CF_EACH_PDE_CHUNK','64')
    if args.action=='plan':
        print(json.dumps(workflow.defaults(),ensure_ascii=False,indent=2));return
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    if args.action in ('run','fields','evaluate') and args.device!='cuda':
        raise RuntimeError('Formal runs require CUDA; use smoke for CPU diagnostics')
    profile='smoke' if args.action=='smoke' else 'formal'
    with lock(out):
        p=workflow.seal(out,profile)
        acceptance_path=out/('acceptance_'+args.device+'.json')
        if not acceptance_path.exists():
            import calibrated_acceptance
            report=calibrated_acceptance.run(out,args.device)
            report['protocol_fingerprint']=p['fingerprint'];dump(acceptance_path,report)
        accepted=read(acceptance_path)
        if not accepted['passed'] or accepted['protocol_fingerprint']!=p['fingerprint'] or accepted['device']!=args.device:
            raise RuntimeError('Numerical acceptance mismatch')
        if args.action=='validate':return
        if args.action in ('run','fields','smoke'):
            for row in p['targets']:
                for seed in p['config']['field_seeds']:
                    status=fields.run(out,row,seed,args.device,args.max_new_steps)
                    if not status['complete']:
                        print('CHECKPOINT_SAVED; rerun the same command/output to resume',flush=True);return
            fields.freeze(out)
            if args.action=='fields':return
        fields.evaluate(out,args.device)
        summary=read(out/'evaluation/summary.json')
        if not summary['complete']:raise RuntimeError('Incomplete evaluation')
        dump(out/'complete.json',dict(complete=True,protocol_fingerprint=p['fingerprint'],
            profile=profile,diagnostic_only=profile=='smoke',cases=[217],seeds=p['config']['field_seeds'],
            budgets=p['config']['field']['budgets'],probability_network_updates=0,
            probability_gate_sha256=sha(out/'probability_frozen.json'),
            field_gate_sha256=sha(out/'field_selection_frozen.json'),
            evaluation_sha256=sha(out/'evaluation/summary.json'),
            acceptance_sha256=sha(acceptance_path),data_role=p['config']['data_role']))
        print('RECONSTRUCTION_COMPLETE',out,flush=True)


if __name__=='__main__':main()
