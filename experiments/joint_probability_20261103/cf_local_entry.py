"""Local entry pinned to the new nu_s=0 / F_ns=1 CF bundle.

`verify` only checks package bytes and metadata. `run` explicitly starts the
unchanged full CF inference protocol on CPU, writing to a separate directory.
This file is not imported by the frozen eighteen-run simulation experiment.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
RUNTIME = BASE/'outputs/JointProbabilityPhysics_20261103/cf_runtime'
EXPECTED_MANIFEST = '5fa36a50a87edb8aababb0dc0a3c27a6e5f71f05bd041cfc1c1e145fe3c550dc'
CASES = ('cf_113_camera100_f0184', 'cf_217_camera100_f0333')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def verify():
    if sha(RUNTIME/'bundle_manifest.json') != EXPECTED_MANIFEST:
        raise RuntimeError('This entry is pinned to a different CF bundle manifest')
    spec = importlib.util.spec_from_file_location('cf_entry_package_check', RUNTIME/'package_tool.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.verify(RUNTIME)
    contract = read(RUNTIME/'SOURCE_CONTRACT.json')
    fingerprint = contract.pop('contract_fingerprint')
    actual = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False).encode('utf8')).hexdigest()
    if fingerprint != actual:
        raise RuntimeError('Source contract fingerprint does not match')
    physics = contract['physics_adaptation']
    if physics['nu_s'] != 0 or physics['Fns_prefactor'] != 1:
        raise RuntimeError('The requested evolution physics is not present')
    return dict(runtime=str(RUNTIME), verified_files=len(manifest['files']),
        bundle_manifest_sha256=EXPECTED_MANIFEST,
        source_contract_sha256=sha(RUNTIME/'SOURCE_CONTRACT.json'),
        source_contract_fingerprint=fingerprint, nu_s_m2_s=0.0,Fns_prefactor=1.0,
        inference_started=False, network_finetuning=False,
        prior='Historical frozen B covariance, not recalibrated for the changed deterministic operator')


@contextmanager
def output_lock(directory):
    """Prevent two instances of this local entry from sharing one output."""
    with (directory/'.cf_local_entry.lock').open('a+b') as stream:
        stream.seek(0)
        if stream.read(1) == b'':
            stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError('Another local CF entry owns this output directory') from error
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run(args, verified):
    destination = args.out.resolve()
    experiment = RUNTIME.parent
    if destination == RUNTIME or destination.is_relative_to(RUNTIME):
        raise ValueError('Use a new results directory outside the immutable runtime')
    if destination == experiment or destination.is_relative_to(experiment/'runs'):
        raise ValueError('Do not mix CF output with the frozen simulation experiment')
    old_contract = destination/'contract.json'
    if old_contract.exists():
        previous = read(old_contract)
        if previous.get('bundle_sha256') != EXPECTED_MANIFEST:
            raise ValueError('Existing output belongs to another CF bundle; use a new directory')
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError('A first run requires an empty output directory')
    destination.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(RUNTIME/'run_cf.py'), '--device', 'cpu', '--allow-full-cpu',
        '--out', str(destination), '--max-seconds', str(args.max_seconds)]
    if args.case:
        command += ['--case', *args.case]
    with output_lock(destination):
        record = dict(verified)
        record.update(inference_started=True,
            started_utc=datetime.now(timezone.utc).isoformat(), command=command,
            output=str(destination), requested_cases=args.case or list(CASES),
            complete_100ms=True, smoke=False, entry_sha256=sha(__file__))
        metadata = destination/'LOCAL_ENTRY.json'
        with (destination/'job.log').open('ab', buffering=0) as log:
            log.write(('\nCF nu_s=0/F_ns=1 local invocation '+record['started_utc']+'\n').encode('utf8'))
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'))
            record['pid'] = process.pid
            metadata.write_text(json.dumps(record,indent=2),encoding='utf8')
            print(json.dumps(record,indent=2),flush=True)
            try:
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        record.update(exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat(),
            status={0:'workflow_complete_check_numerical_results',3:'numerical_gate_blocked',
                75:'budget_boundary_resumable'}.get(code,'error'))
        metadata.write_text(json.dumps(record,indent=2),encoding='utf8')
        print(json.dumps(record,indent=2),flush=True)
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify','run'))
    parser.add_argument('--out',type=Path,help='New CF results directory; required for run')
    parser.add_argument('--max-seconds',type=float,default=3600)
    parser.add_argument('--case',nargs='+',choices=CASES)
    args=parser.parse_args()
    if args.action=='run' and args.out is None:
        parser.error('run requires --out')
    if args.max_seconds<=0:
        parser.error('--max-seconds must be positive')
    verified=verify()
    if args.action=='verify':
        print(json.dumps(verified,indent=2));return
    raise SystemExit(run(args,verified))


if __name__=='__main__':
    main()
