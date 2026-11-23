"""One locked local pipeline; all selections precede target evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess
import sys
import os
import time
import json

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[2]/'outputs/PINNTwoFluid_20261105'

def read(path):return json.loads(path.read_text(encoding='utf8'))
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8');tmp.replace(path)

def batch(jobs,workers,fp):
    pending=[];finished=[];active=[];failed=[];begun=time.perf_counter()
    for job in jobs:
        if job['status'].exists() and read(job['status']).get('complete'):
            saved=read(job['status']);assert saved['protocol_fingerprint']==fp
            finished.append(dict(label=job['label'],reused_completed=True))
        else:pending.append(job)
    def status():
        dump(OUT/'inversion_batch_status.json',dict(complete=len(finished)==len(jobs) and not failed,
            pending=[j['label'] for j in pending],active=[dict(label=j['label'],pid=p.pid) for j,p,s in active],
            finished=finished,failed=failed,wall_seconds=time.perf_counter()-begun,
            updated_utc=datetime.now(timezone.utc).isoformat()))
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
    from pinn_common import verify_protocol,SEEDS,CASES,ARMS,case_name
    p=verify_protocol();(OUT/'logs').mkdir(parents=True,exist_ok=True)
    lock=OUT/'pipeline.lock'
    with lock.open('x',encoding='utf8') as s:s.write(str(os.getpid()))
    start=time.perf_counter();records=[]
    def status(stage,complete=False,failed=False):
        dump(OUT/'pipeline_status.json',dict(stage=stage,complete=complete,failed=failed,pid=os.getpid(),
            protocol_fingerprint=p['fingerprint'],records=records,elapsed_seconds=time.perf_counter()-start,
            updated_utc=datetime.now(timezone.utc).isoformat()))
    def command(label,script):
        status(label);t=time.perf_counter()
        with (OUT/'logs'/(label+'.log')).open('a',encoding='utf8') as stream:
            rc=subprocess.run([sys.executable,*script],stdout=stream,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).returncode
        records.append(dict(stage=label,exit_code=rc,seconds=time.perf_counter()-t))
        if rc:raise RuntimeError(label+' failed; inspect log')
    try:
        jobs=[]
        for seed in SEEDS:
            for U in CASES:
                for arm in ARMS:
                    label=f'{case_name(U)}_{seed}_{arm}'
                    jobs.append(dict(label=label,status=OUT/'runs'/case_name(U)/arm/f'seed_{seed}'/'status.json',
                        args=[str(HERE/'experiment.py'),'run','--U',str(U),'--seed',str(seed),'--arm',arm]))
        status('inversion');batch(jobs,p['inversion_workers'],p['fingerprint'])
        command('freeze_fields',[str(HERE/'experiment.py'),'freeze'])
        for label,script in [('evaluation','evaluate_pinn.py'),('plots','plot_pinn.py'),('audit','audit_results.py')]:
            command(label,[str(HERE/script)])
        status('complete',complete=True);print('PINN_PIPELINE_COMPLETE',flush=True)
    except BaseException as error:
        status(str(error),failed=True);raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
