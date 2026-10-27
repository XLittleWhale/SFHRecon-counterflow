"""Source-only formal release checks; no private research data needed."""
import hashlib
import json
from pathlib import Path
import pytest
from counterflow.io import sha
from counterflow_full import core
from counterflow_full.verify import verify

ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'artifacts/fullwindow_4mm_20261026'


def test_frozen_baseline_replays_from_initial_state():
    result=verify(BASELINE)
    assert result['files_checked']==100
    assert result['window_vector_RMS_mm_s']==pytest.approx([.489406132,.891792802],abs=1e-8)


def test_published_scientific_kernels_match_returned_run():
    policy=json.loads((BASELINE/'evidence/policy.json').read_text(encoding='utf8'))
    # Git normalizes line endings; compare canonical LF text against the original
    # package's copied hashes using the archived canonical hash list.
    manifest=json.loads((Path(core.__file__).parent/'resources/scientific_kernels_sha256.json').read_text(encoding='utf8'))
    for name,digest in manifest.items():
        actual=(ROOT/'src'/name).read_text(encoding='utf8').encode('utf8')
        assert hashlib.sha256(actual).hexdigest()==digest['lf_sha256'],name
        assert digest['original_sha256']==policy['source_hashes'][name]


def test_portable_builder_resources_match_baseline():
    resources=Path(core.__file__).parent/'resources'
    assert sha(resources/'operator.pt')==sha(BASELINE/'operator.pt')
    assert sha(resources/'physics_4mm.json')==sha(BASELINE/'physics.json')
    assert core.config()==json.loads((BASELINE/'config.json').read_text(encoding='utf8'))
