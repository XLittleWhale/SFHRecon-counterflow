"""Run the authorized read-only attribution after the entire main pipeline."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "outputs/SharedTransition_20261104"


def main():
    started = time.perf_counter()
    def status(stage, complete=False, failed=False, exit_code=None):
        value = dict(stage=stage, complete=complete, failed=failed,
                     exit_code=exit_code, elapsed_seconds=time.perf_counter()-started,
                     updated_utc=datetime.now(timezone.utc).isoformat())
        path = OUT / "auxiliary_pipeline_status.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2), encoding="utf8")
        temp.replace(path)

    status("waiting_for_main_pipeline")
    while True:
        pipeline = json.loads((OUT / "pipeline_status.json").read_text(encoding="utf8"))
        if pipeline.get("failed"):
            status("main_pipeline_failed", failed=True)
            raise RuntimeError("Main pipeline failed; auxiliary scoring not started")
        if pipeline.get("complete"):
            break
        time.sleep(5)
    status("attribution")
    with (OUT / "logs/auxiliary_probability_attribution.log").open("a", encoding="utf8") as stream:
        rc = subprocess.run([sys.executable, str(HERE / "auxiliary_probability_attribution.py")],
            stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode
    status("complete" if rc == 0 else "attribution_failed", complete=rc == 0, failed=rc != 0, exit_code=rc)
    if rc:
        raise RuntimeError("Auxiliary attribution failed; inspect its log")
    print("AUXILIARY_PIPELINE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
