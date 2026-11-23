"""Single-owner local pipeline; a stale lock never authorizes another worker."""
from cf_common import *
import subprocess
import ctypes

def pid_alive(pid):
    if not pid or pid == os.getpid(): return False
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return ctypes.get_last_error() == 5
    try:
        code = ctypes.c_ulong()
        return not kernel.GetExitCodeProcess(handle,ctypes.byref(code)) or code.value == 259
    finally:
        kernel.CloseHandle(handle)

def live_workers():
    # Check several independent persisted process records, even if a lock vanished.
    # A recycled live PID is treated conservatively and requires manual inspection.
    candidates = {}
    paths = [OUT/'pipeline_status.json',OUT/'pipeline.lock',OUT/'inversion_batch_status.json',
             OUT/'postprocess_status.json']
    paths += list((OUT/'runs').glob('*/*/status.json'))+list((OUT/'runs').glob('*/*/run.lock'))
    for path in paths:
        if not path.exists(): continue
        record = read(path)
        if not record.get('complete') and record.get('pid'):
            candidates[int(record['pid'])] = str(path)
        for item in record.get('active',[]):
            candidates[int(item['pid'])] = str(path)
    return [dict(pid=pid,record=path) for pid,path in candidates.items() if pid_alive(pid)]

def batch(jobs, workers, fp):
    pending, finished, active, failed = [], [], [], []
    begun = time.perf_counter()
    for job in jobs:
        if job['status'].exists() and read(job['status']).get('complete'):
            assert read(job['status'])['protocol_fingerprint'] == fp
            finished.append(dict(label=job['label'],reused_completed=True))
        else:
            pending.append(job)

    def status():
        dump(OUT/'inversion_batch_status.json',dict(
            complete=len(finished)==len(jobs) and not failed,
            pending=[j['label'] for j in pending],
            active=[dict(label=j['label'],pid=proc.pid) for j,proc,s in active],
            finished=finished,failed=failed,wall_seconds=time.perf_counter()-begun,updated_utc=utc()))

    while pending or active:
        while pending and len(active)<workers and not failed:
            job = pending.pop(0)
            stream = (OUT/'logs'/(job['label']+'.log')).open('a',encoding='utf8')
            proc = subprocess.Popen([sys.executable,*job['args']],stdout=stream,stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((job,proc,stream))
            print('START',job['label'],proc.pid,flush=True)
        for item in active.copy():
            job,proc,stream = item
            rc = proc.poll()
            if rc is not None:
                stream.close()
                active.remove(item)
                record = dict(label=job['label'],exit_code=rc)
                if rc==0 and job['status'].exists() and read(job['status']).get('complete'):
                    finished.append(record)
                else:
                    failed.append(record)
                print('FINISH',record,flush=True)
        status()
        if failed and not active:
            raise RuntimeError('CF batch failed; inspect logs and persisted checkpoints: '+str(failed))
        if pending or active:
            time.sleep(2)
    status()

def main():
    p = verify_protocol()
    conflicts = live_workers()
    if conflicts:
        raise RuntimeError('Another CF pipeline/worker is alive: '+str(conflicts))
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    lock = OUT/'pipeline.lock'
    with lock.open('x',encoding='utf8') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(),started_utc=utc())))
    started = time.perf_counter()
    records = []

    def status(stage, complete=False, failed=False):
        dump(OUT/'pipeline_status.json',dict(stage=stage,complete=complete,failed=failed,
            pid=os.getpid(),protocol_fingerprint=p['fingerprint'],records=records,
            elapsed_seconds=time.perf_counter()-started,updated_utc=utc()))

    def command(label,args):
        status(label)
        begin = time.perf_counter()
        with (OUT/'logs'/(label+'.log')).open('a',encoding='utf8') as stream:
            proc = subprocess.Popen([sys.executable,*args],stdout=stream,stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            dump(OUT/'postprocess_status.json',dict(stage=label,pid=proc.pid,parent_pid=os.getpid(),
                complete=False,args=args,protocol_fingerprint=p['fingerprint'],started_utc=utc()))
            rc = proc.wait()
        dump(OUT/'postprocess_status.json',dict(stage=label,pid=proc.pid,parent_pid=os.getpid(),
            complete=rc==0,exit_code=rc,args=args,protocol_fingerprint=p['fingerprint'],finished_utc=utc()))
        records.append(dict(stage=label,exit_code=rc,seconds=time.perf_counter()-begin))
        if rc:
            raise RuntimeError(label+' failed; inspect log')
    try:
        jobs = []
        for seed in SEEDS:
            for case in CASES:
                label = f'{case}_{seed}'
                jobs.append(dict(label=label,status=OUT/'runs'/case/f'seed_{seed}'/'status.json',
                    args=[str(HERE/'cf_experiment.py'),'run','--case',case,'--seed',str(seed)]))
        status('inversion')
        batch(jobs,p['inversion_workers'],p['fingerprint'])
        command('freeze_fields',[str(HERE/'cf_experiment.py'),'freeze'])
        for label,script in [('evaluation','cf_evaluate.py'),('plots','cf_plot.py'),('audit','audit_results.py')]:
            command(label,[str(HERE/script)])
        status('complete',complete=True)
        print('PINNCF_PIPELINE_COMPLETE',flush=True)
    except BaseException as error:
        status(str(error),failed=True)
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__ == '__main__':
    setup()
    main()
