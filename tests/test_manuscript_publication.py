"""The published result snapshot must match the figures and cases selected by the paper."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_result_snapshot():
    manifest = json.loads((ROOT / "config/manuscript_baseline.json").read_text(encoding="utf-8"))
    output = ROOT / "results" / manifest["id"]
    prefix = manifest["paper"] + "/result_figures/"
    filenames = {p.name for p in output.glob("*.pdf")}
    assert len(filenames) == 7
    for name in filenames | {"metrics.json", "performance_table.tex", "reference_cases_table.tex"}:
        actual = hashlib.sha256((output / name).read_bytes()).hexdigest()
        assert actual == manifest["workspace_files"][prefix + name]["sha256"], name
    published = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert [row["case"] for row in published] == [case["case"] for case in manifest["cases"]]
    for row, case in zip(published, manifest["cases"]):
        for key in ("selected_steps", "H_nll_per_ms", "vector_rmse_mm_s"):
            assert row[key] == case[key]
