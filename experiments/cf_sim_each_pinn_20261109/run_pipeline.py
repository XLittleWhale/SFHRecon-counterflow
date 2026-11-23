"""Three CPU workers: each trains, freezes, reconstructs; final evaluation is gated."""
from common import *
import argparse
import subprocess
import psutil

def process_alive(record):
    pid=record.get('pid')
    if not pid: return False
    try:
        proc=psutil.Process(int(pid))
        # Old lock records have no creation time: a reused live PID is treated conservatively.
        return 'process_created' not in record or abs(proc.create_time()-record['process_created'])<.01
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True

def acquire_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        record=read(path)
        if process_alive(record):
            raise RuntimeError('Another live owner holds '+str(path)+': '+str(record))
        # Preserve the old process record rather than silently discarding it.
        path.rename(path.with_name(path.name+'.stale.'+uuid.uuid4().hex))
    with path.open('x',encoding='utf8') as stream:
        json.dump(dict(pid=os.getpid(),process_created=psutil.Process().create_time(),started_utc=utc()),stream)

def recover_dead_field_lock(case,seed):
    path=field_folder(case,seed)/'run.lock'
    if path.exists():
        if process_alive(read(path)):
            raise RuntimeError('A field process is already alive: '+str(path))
        path.rename(path.with_name(path.name+'.stale.'+uuid.uuid4().hex))

def worker(case,seed):
    p=verify_protocol()
    folder=OUT/'jobs'/case/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    lock=folder/'job.lock';acquire_lock(lock)
    def status(stage,complete=False,error=None):
        dump(folder/'status.json',dict(case=case,seed=seed,stage=stage,complete=complete,
            pid=os.getpid(),process_created=psutil.Process().create_time(),error=error,
            protocol_fingerprint=p['fingerprint'],updated_utc=utc()))
    try:
        import train_probability as T
        import reconstruct as R
        status('probability_training')
        T.run(prepared(case),probability_folder(case,seed),TRAIN_SEEDS[SEEDS.index(seed)],p['probability'])
        status('field_reconstruction')
        recover_dead_field_lock(case,seed)
        result=R.run(case,seed)
        assert result['complete']
        status('complete',True)
    except BaseException as error:
        status('failed',error=repr(error))
        raise
    finally:
        lock.unlink(missing_ok=True)

def main():
    p=verify_protocol()
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    lock=OUT/'pipeline.lock';acquire_lock(lock)
    pending=[];finished=[];failed=[];active=[]
    started=time.perf_counter()
    def report(stage,complete=False):
        progress=[]
        for job,proc,stream in active:
            case,seed=job
            row=dict(case=case,seed=seed,pid=proc.pid)
            for key,path in [('job',OUT/'jobs'/case/f'seed_{seed}'/'status.json'),
                             ('probability',probability_folder(case,seed)/'status.json'),
                             ('field',field_folder(case,seed)/'status.json')]:
                if path.exists(): row[key]=read(path)
            progress.append(row)
        dump(OUT/'pipeline_status.json',dict(stage=stage,complete=complete,failed=failed,
            pid=os.getpid(),process_created=psutil.Process().create_time(),active=progress,
            pending=[dict(case=c,seed=s) for c,s in pending],finished=finished,
            elapsed_seconds=time.perf_counter()-started,updated_utc=utc(),
            protocol_fingerprint=p['fingerprint']))
    try:
        for seed in SEEDS:
            for case in CASES:
                folder=OUT/'jobs'/case/f'seed_{seed}'
                job_lock=folder/'job.lock'
                if job_lock.exists() and process_alive(read(job_lock)):
                    raise RuntimeError('Orphaned live worker: '+str(job_lock))
                status_path=folder/'status.json'
                if status_path.exists() and read(status_path).get('complete'):
                    status=read(status_path)
                    assert status['protocol_fingerprint']==p['fingerprint']
                    finished.append(dict(case=case,seed=seed,reused_completed=True))
                else: pending.append((case,seed))
        while pending or active:
            while pending and len(active)<p['execution']['workers'] and not failed:
                job=pending.pop(0);case,seed=job
                stream=(OUT/'logs'/f'{case}_{seed}.log').open('a',encoding='utf8')
                env=dict(os.environ,PYTHONUNBUFFERED='1',CF_EACH_THREADS=str(p['execution']['threads_per_worker']))
                for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
                    env[key]=str(p['execution']['threads_per_worker'])
                proc=subprocess.Popen([sys.executable,str(HERE/'run_pipeline.py'),'worker',
                    '--case',case,'--seed',str(seed)],stdout=stream,stderr=subprocess.STDOUT,
                    env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active.append((job,proc,stream))
                print('START',case,seed,proc.pid,flush=True)
            for item in active.copy():
                job,proc,stream=item
                rc=proc.poll()
                if rc is not None:
                    stream.close();active.remove(item)
                    case,seed=job;status_path=OUT/'jobs'/case/f'seed_{seed}'/'status.json'
                    row=dict(case=case,seed=seed,exit_code=rc)
                    if rc==0 and status_path.exists() and read(status_path).get('complete'): finished.append(row)
                    else: failed.append(row)
                    print('FINISH',row,flush=True)
            report('training_and_reconstruction')
            if failed and not active:
                raise RuntimeError('Worker failed; inspect per-job log/checkpoint: '+str(failed))
            if pending or active: time.sleep(5)
        report('freeze_all_selections')
        import reconstruct
        reconstruct.freeze()
        report('heldout_and_reference_evaluation')
        import evaluate
        evaluate.main()
        report('complete',True)
        print('CF_SIM_EACH_PIPELINE_COMPLETE',flush=True)
    except BaseException as error:
        report('failed: '+repr(error))
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=('run','worker'),default='run',nargs='?')
    ap.add_argument('--case',choices=CASES,default=CASES[0])
    ap.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    args=ap.parse_args();setup()
    if args.action=='worker': worker(args.case,args.seed)
    else: main()
