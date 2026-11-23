"""Bounded local workers for the two explicitly frozen experiment stages."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2]/'outputs/SharedTransition_20261104'
ARMS = ('cap_retrain', 'independent', 'shared_gru')


def main(stage, workers):
    if not 1 <= workers <= 4:
        raise ValueError('Use one to four local workers')
    if stage == 'training':
        import training as T
        T.verify_protocol()
        jobs = [dict(arm=arm, seed=seed) for seed in T.SEEDS for arm in ARMS]
        def status_path(j):
            return OUT/'training'/j['arm']/f"seed_{j['seed']}"/'status.json'
        def command(j):
            return [sys.executable, str(HERE/'training.py'), 'run', '--arm', j['arm'], '--seed', str(j['seed'])]
    else:
        import experiment as E
        E.verify_protocol()
        jobs = [dict(U=U, case=E.case_name(U), seed=seed, arm=arm)
                for seed in E.SEEDS for U in E.CASES for arm in ARMS]
        def status_path(j):
            return OUT/'runs'/j['case']/f"seed_{j['seed']}"/j['arm']/'status.json'
        def command(j):
            return [sys.executable, str(HERE/'experiment.py'), 'run', '--U', str(j['U']),
                    '--arm', j['arm'], '--seed', str(j['seed'])]
    pending, finished, active, failed = [], [], [], []
    for job in jobs:
        path = status_path(job)
        if path.exists() and json.loads(path.read_text(encoding='utf8')).get('complete'):
            finished.append(dict(job, already_complete=True))
        else:
            pending.append(job)
    logs = OUT/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    def persist():
        value = dict(stage=stage, workers=workers, updated_utc=datetime.now(timezone.utc).isoformat(),
            pending=pending, active=[dict(j, pid=p.pid) for j, p, stream in active],
            finished=finished, failed=failed, complete=len(finished)==len(jobs) and not failed,
            elapsed_seconds=time.perf_counter()-started,
            timing_scope='This invocation; per-run complete times are stored separately')
        path = OUT/f'{stage}_batch_status.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
        tmp.replace(path)
    while pending or active:
        while pending and len(active) < workers and not failed:
            job = pending.pop(0)
            label = '_'.join([stage, job.get('case', 'network'), str(job['seed']), job['arm']])
            stream = (logs/(label+'.log')).open('a', encoding='utf8')
            process = subprocess.Popen(command(job), stdout=stream, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            active.append((job, process, stream))
            print('START', job, 'pid', process.pid, flush=True)
        for item in active.copy():
            job, process, stream = item
            rc = process.poll()
            if rc is not None:
                stream.close()
                active.remove(item)
                value = dict(job, exit_code=rc)
                if rc == 0 and status_path(job).exists() and json.loads(status_path(job).read_text(encoding='utf8')).get('complete'):
                    finished.append(value)
                else:
                    failed.append(value)
                print('FINISH', value, flush=True)
        persist()
        if failed and not active:
            raise RuntimeError('A stage job failed; see logs and batch status. No dependent stage was started.')
        if pending or active:
            time.sleep(2)
    persist()
    print(stage.upper()+'_BATCH_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('training', 'inversion'), required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    main(args.stage, args.workers)
