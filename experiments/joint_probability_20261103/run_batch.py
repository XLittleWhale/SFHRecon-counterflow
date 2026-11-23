"""Bounded local process pool; each job resumes its own saved optimizer state."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import experiment as E

def main(workers):
    E.verify_protocol()
    logdir=E.OUT/'logs';logdir.mkdir(parents=True,exist_ok=True)
    pending=[(U,seed,arm) for seed in E.SEEDS for U in E.CASES for arm in E.ARMS]
    active=[];finished=[];failed=[]
    started=time.perf_counter()
    def identity(job):
        U,seed,arm=job
        return dict(U=U,case=E.case_name(U),seed=seed,arm=arm)
    def update():
        E.dump(E.OUT/'batch_status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),
            workers=workers,pending=[identity(j) for j in pending],
            active=[dict(**identity(j),pid=p.pid) for j,p,h in active],
            finished=finished,failed=failed,elapsed_seconds=time.perf_counter()-started,
            complete=not pending and not active and not failed))
    while pending or active:
        while pending and len(active)<workers:
            job=pending.pop(0);U,seed,arm=job
            dest=E.OUT/'runs'/E.case_name(U)/f'seed_{seed}'/arm
            status=dest/'status.json'
            if status.exists() and json.loads(status.read_text())['complete']:
                finished.append(dict(**identity(job),resumed_completed=True));continue
            logfile=logdir/f'{E.case_name(U)}_{seed}_{arm}.log'
            handle=logfile.open('a',encoding='utf8')
            cmd=[sys.executable,'-u',str(E.HERE/'experiment.py'),'run','--U',str(U),'--seed',str(seed),'--arm',arm]
            process=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,
                cwd=E.BASE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((job,process,handle));print('START',identity(job),process.pid,flush=True)
        remaining=[]
        for job,process,handle in active:
            code=process.poll()
            if code is None:remaining.append((job,process,handle));continue
            handle.close()
            record=dict(**identity(job),exit_code=code)
            (finished if code==0 else failed).append(record)
            print('FINISH',record,flush=True)
        active=remaining;update()
        if active:time.sleep(5)
    update()
    if failed:raise SystemExit('One or more fits failed; inspect per-run logs.')
    print('ALL_FITS_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if not 1<=args.workers<=4:raise ValueError('Local workers must be 1..4')
    main(args.workers)
