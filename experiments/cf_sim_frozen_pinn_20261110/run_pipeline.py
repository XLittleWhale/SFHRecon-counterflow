"""Sequential GPU field fits; the probability model is never optimized."""
from common import *
import argparse
import subprocess
import psutil

def process_alive(record):
    try:
        proc=psutil.Process(int(record.get('pid',0)))
        return 'process_created' not in record or abs(proc.create_time()-record['process_created'])<.01
    except (psutil.NoSuchProcess,ValueError): return False
    except psutil.AccessDenied: return True

def acquire_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if process_alive(read(path)): raise RuntimeError('Another live process holds '+str(path))
        path.rename(path.with_name(path.name+'.stale.'+uuid.uuid4().hex))
    with path.open('x',encoding='utf8') as stream:
        json.dump(dict(pid=os.getpid(),process_created=psutil.Process().create_time(),started_utc=utc()),stream)

def worker(case,seed):
    p=verify_protocol()
    folder=OUT/'jobs'/case/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    lock=folder/'job.lock';acquire_lock(lock)
    def status(stage,complete=False,error=None):
        dump(folder/'status.json',dict(case=case,seed=seed,stage=stage,complete=complete,
            pid=os.getpid(),process_created=psutil.Process().create_time(),error=error,
            protocol_fingerprint=p['fingerprint'],probability_updates=0,updated_utc=utc()))
    try:
        import reconstruct as R
        status('field_reconstruction_with_Nov5_frozen_probability')
        field_lock=field_folder(case,seed)/'run.lock'
        if field_lock.exists():
            if process_alive(read(field_lock)): raise RuntimeError('A field worker is already alive')
            field_lock.rename(field_lock.with_name(field_lock.name+'.stale.'+uuid.uuid4().hex))
        result=R.run(case,seed)
        assert result['complete']
        status('complete',True)
    except BaseException as error:
        status('failed',error=repr(error));raise
    finally: lock.unlink(missing_ok=True)

def main():
    p=verify_protocol()
    if device()!=p['execution']['device']: raise RuntimeError('Device differs from protocol')
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    lock=OUT/'pipeline.lock';acquire_lock(lock)
    started=time.perf_counter();finished=[]
    def report(stage,active=None,complete=False,error=None):
        dump(OUT/'pipeline_status.json',dict(stage=stage,complete=complete,error=error,
            pid=os.getpid(),process_created=psutil.Process().create_time(),active=active,finished=finished,
            total_runs=len(CASES)*len(SEEDS),elapsed_seconds=time.perf_counter()-started,
            protocol_fingerprint=p['fingerprint'],updated_utc=utc()))
    try:
        for seed in SEEDS:
            for case in CASES:
                job=OUT/'jobs'/case/f'seed_{seed}'
                if (job/'job.lock').exists() and process_alive(read(job/'job.lock')):
                    raise RuntimeError('A prior worker is still alive: '+str(job))
                if (job/'status.json').exists() and read(job/'status.json').get('complete'):
                    assert read(job/'status.json')['protocol_fingerprint']==p['fingerprint']
                    finished.append(dict(case=case,seed=seed,reused=True));continue
                with (OUT/'logs'/f'{case}_{seed}.log').open('a',encoding='utf8') as stream:
                    proc=subprocess.Popen([sys.executable,'-u',str(HERE/'run_pipeline.py'),'worker',
                        '--case',case,'--seed',str(seed)],stdout=stream,stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    print('START_FROZEN_FIELD',case,seed,proc.pid,flush=True)
                    while proc.poll() is None:
                        info=dict(case=case,seed=seed,pid=proc.pid)
                        field_status=field_folder(case,seed)/'status.json'
                        if field_status.exists(): info['field']=read(field_status)
                        report('field_reconstruction',info);time.sleep(5)
                if proc.returncode or not read(job/'status.json').get('complete'):
                    raise RuntimeError('Field failed: '+case+'; inspect its log')
                finished.append(dict(case=case,seed=seed,exit_code=proc.returncode))
        report('freeze_selections')
        import reconstruct
        reconstruct.freeze()
        report('numeric_evaluation')
        import evaluate
        evaluate.main()
        report('complete',complete=True)
        print('FROZEN_COMPARISON_COMPLETE',flush=True)
    except BaseException as error:
        report('failed',error=repr(error));raise
    finally: lock.unlink(missing_ok=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('run','worker'),default='run',nargs='?')
    ap.add_argument('--case',choices=CASES,default=CASES[0]);ap.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    args=ap.parse_args();setup()
    if args.action=='worker': worker(args.case,args.seed)
    else: main()
