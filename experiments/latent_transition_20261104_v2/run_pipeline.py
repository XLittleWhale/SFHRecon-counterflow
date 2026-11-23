"""Resumable bounded local pipeline; never starts target scoring before gates."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess
import sys
import json
import os
import time

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[2]/'outputs/LatentTransition_20261104_v2'

def read(path):return json.loads(path.read_text(encoding='utf8'))
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8');tmp.replace(path)

def batch(stage,jobs,workers=4):
    pending=[];finished=[];active=[];failed=[];begun=time.perf_counter()
    for job in jobs:
        if job['status'].exists() and read(job['status']).get('complete'):finished.append(dict(label=job['label'],reused_completed=True))
        else:pending.append(job)
    def status():
        dump(OUT/(stage+'_batch_status.json'),dict(stage=stage,complete=len(finished)==len(jobs) and not failed,
            pending=[j['label'] for j in pending],active=[dict(label=j['label'],pid=p.pid) for j,p,s in active],
            finished=finished,failed=failed,wall_seconds=time.perf_counter()-begun,updated_utc=datetime.now(timezone.utc).isoformat()))
    while pending or active:
        while pending and len(active)<workers and not failed:
            j=pending.pop(0);stream=(OUT/'logs'/(j['label']+'.log')).open('a',encoding='utf8')
            proc=subprocess.Popen([sys.executable,*j['args']],stdout=stream,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((j,proc,stream));print('START',j['label'],proc.pid,flush=True)
        for item in active.copy():
            j,proc,stream=item;rc=proc.poll()
            if rc is not None:
                stream.close();active.remove(item);record=dict(label=j['label'],exit_code=rc)
                if rc==0 and j['status'].exists() and read(j['status']).get('complete'):finished.append(record)
                else:failed.append(record)
                print('FINISH',record,flush=True)
        status()
        if failed and not active:raise RuntimeError('Batch failed: '+str(failed))
        if pending or active:time.sleep(2)
    status()

def main():
    import train_latent as T
    T.verify_protocol()
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    lock=OUT/'pipeline.lock'
    with lock.open('x',encoding='utf8') as s:s.write(str(os.getpid()))
    start=time.perf_counter();records=[]
    def status(stage,complete=False,failed=False):
        dump(OUT/'pipeline_status.json',dict(stage=stage,complete=complete,failed=failed,pid=os.getpid(),records=records,
            elapsed_seconds=time.perf_counter()-start,updated_utc=datetime.now(timezone.utc).isoformat()))
    def command(label,*args):
        status(label);t=time.perf_counter()
        with (OUT/'logs'/(label+'.log')).open('a',encoding='utf8') as stream:
            rc=subprocess.run([sys.executable,*map(str,args)],stdout=stream,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).returncode
        records.append(dict(stage=label,exit_code=rc,seconds=time.perf_counter()-t))
        if rc:raise RuntimeError(label+' failed; inspect log')
    try:
        status('training')
        jobs=[dict(label='train_'+str(seed),status=OUT/'training'/f'seed_{seed}'/'status.json',
                   args=[str(HERE/'train_latent.py'),'run','--seed',str(seed)]) for seed in T.TRAIN_SEEDS]
        batch('training',jobs,3)
        command('freeze_heads',HERE/'train_latent.py','freeze')
        command('reuse_iid',HERE/'experiment_latent.py','reuse')
        status('inversion');jobs=[]
        for seed in T.SEEDS:
            for U in T.CASES:
                case=T.case_name(U)
                for arm in T.NEW_ARMS:
                    jobs.append(dict(label=f'inversion_{case}_{seed}_{arm}',
                        status=OUT/'runs'/case/f'seed_{seed}'/arm/'status.json',
                        args=[str(HERE/'experiment_latent.py'),'run','--U',str(U),'--seed',str(seed),'--arm',arm]))
        batch('inversion',jobs,4)
        command('freeze_fields',HERE/'experiment_latent.py','freeze')
        command('evaluate_and_plot',HERE/'evaluate_latent.py')
        status('complete',complete=True);print('LATENT_TRANSITION_PIPELINE_COMPLETE',flush=True)
    except BaseException as error:
        status(str(error),failed=True);raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':main()
