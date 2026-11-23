"""Continue this authorized local experiment through frozen evaluation and plots.

An already-running training batch is allowed to finish. Any failed stage stops
the pipeline, preserving its log; no acceptance rules or budgets are changed.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2]/'outputs/SharedTransition_20261104'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    started = time.perf_counter()
    records = []
    def status(stage, complete=False, failed=False):
        value = dict(stage=stage, complete=complete, failed=failed, records=records,
                     elapsed_seconds=time.perf_counter()-started,
                     updated_utc=datetime.now(timezone.utc).isoformat())
        path = OUT/'pipeline_status.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2), encoding='utf8')
        temp.replace(path)
    status('waiting_for_nine_training_runs')
    print('PIPELINE waiting for all nine training runs', flush=True)
    while True:
        batch = read(OUT/'training_batch_status.json')
        if batch.get('failed'):
            status('training_failed', failed=True)
            raise RuntimeError('Training failed; pipeline stopped')
        if batch.get('complete'):
            break
        time.sleep(5)
    commands = [
        ('freeze_networks', 'training.py', 'freeze-selections'),
        ('prepare_field_protocol', 'experiment.py', 'prepare'),
        ('field_inversions', 'run_batch.py', '--stage', 'inversion', '--workers', '4'),
        ('freeze_fields', 'experiment.py', 'freeze'),
        ('evaluate', 'evaluate_transition.py'),
        ('plot', 'plot_transition.py'),
    ]
    for name, script, *args in commands:
        status(name)
        print('PIPELINE START', name, flush=True)
        log_path = OUT/'logs'/('pipeline_'+name+'.log')
        t = time.perf_counter()
        with log_path.open('a', encoding='utf8') as stream:
            rc = subprocess.run([sys.executable, str(HERE/script), *args],
                stdout=stream, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).returncode
        records.append(dict(stage=name, exit_code=rc, seconds=time.perf_counter()-t, log=str(log_path)))
        if rc:
            status(name, failed=True)
            raise RuntimeError(f'{name} failed with {rc}; inspect {log_path}')
        print('PIPELINE FINISH', name, flush=True)
    status('complete', complete=True)
    print('SHARED_TRANSITION_PIPELINE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
