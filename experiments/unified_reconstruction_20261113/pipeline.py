"""Full each supervised continuation -> freeze -> PINN -> freeze -> evaluation."""
import argparse
from runtime import *
import method_contract as contract
import workflow

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','audit','prepare','validate','run','train','probability','fields','evaluate','smoke'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=BASE/'outputs/cf_reference_transient1s_50frames_results_20261108')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--profile',choices=['baseline_retrain_all','continuation_short','joint_continuation'],default='baseline_retrain_all')
    parser.add_argument('--max-new-epochs',type=int)
    parser.add_argument('--max-new-steps',type=int)
    args=parser.parse_args(argv);out=args.output.resolve()
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    if args.action=='plan':
        print(json.dumps(dict(config=workflow.defaults(args.profile),readiness=contract.training_readiness()),ensure_ascii=False,indent=2));return
    if args.action in ('audit','prepare'):
        import labels,clock_contract,force_context
        solver=BASE/'reference_solver'
        if not solver.exists():solver=BASE/'outputs/CFReference_2K_20261106/CFReference_2K_AutoDL_Galantucci_1s_50frames_20261108'
        with lock(out):
            for q in (113,175,217):
                case=f'cf_2K_Q{q}';prepared=BASE/'outputs/CFSimEachPINN_20261109/prepared'/case;folder=out/case
                labels.prepare(args.source/case,prepared,solver,folder)
                obs=load(prepared/'probability_observations.pt')
                dump(folder/'window_audit.json',clock_contract.audit_windows(obs['z'],obs['meta']['camera_fps']))
                if args.action=='prepare':force_context.prepare(args.source/case,prepared,folder,args.device)
        return
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    if args.action not in ('validate','smoke') and args.device!='cuda':raise RuntimeError('Use smoke for CPU end-to-end checks; formal runs use CUDA')
    profile='smoke' if args.action=='smoke' else args.profile
    with lock(out):
        if args.action=='validate':
            import acceptance
            dump(out/'validation.json',acceptance.run(args.device));return
        p=workflow.seal(out,profile)
        path=out/'acceptance.json'
        if not path.exists():
            import acceptance
            report=acceptance.run(args.device,tuple(p['config']['cases']))
            report['protocol_fingerprint']=p['fingerprint'];dump(path,report)
        contract.require_training(out,args.device)
        if args.action in ('run','train','probability','smoke'):
            result=workflow.train_probabilities(out,args.device,args.max_new_epochs)
            if not result['complete'] or args.action=='probability':return
        if args.action in ('run','train','fields','smoke'):
            import fields
            for row in p['targets']:
                for seed in p['config']['field_seeds']:
                    status=fields.run(out,row,seed,args.device,args.max_new_steps)
                    if not status['complete']:return
            fields.freeze(out)
            if args.action=='fields':return
        if args.action in ('run','train','evaluate','smoke'):
            import fields
            fields.evaluate(out,args.device)
            dump(out/'complete.json',dict(complete=True,protocol_fingerprint=p['fingerprint'],profile=profile,
                diagnostic_only=profile=='smoke',cases=p['config']['cases'],seeds=p['config']['field_seeds'],
                probability_gate_sha256=sha(out/'probability_frozen.json'),field_gate_sha256=sha(out/'field_selection_frozen.json'),
                evaluation_sha256=sha(out/'evaluation/summary.json')))
            print('END_TO_END_COMPLETE',out,flush=True)

if __name__=='__main__':main()
