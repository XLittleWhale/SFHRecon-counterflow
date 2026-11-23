"""Use already available, verified offline releases without creating another copy."""
import hashlib
import importlib
import json
import sys

from .paths import SECTIONS, install_frozen_imports, read_contained

FULL = SECTIONS[4] / "dependencies/build_inputs/unified_cf/stage/a19a8aeb/UnifiedCF_FullBaseline_GPU_20261113"
CALIBRATED = SECTIONS[4] / "dependencies/build_inputs/cf217_build/cce98ee5/package"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify_bundle(bundle):
    manifest = bundle / "bundle_manifest.json"
    for name, expected in json.loads(manifest.read_text(encoding="utf-8"))["sha256"].items():
        path = (bundle / name).resolve(strict=True)
        if not path.is_relative_to(bundle) or sha(path) != expected:
            raise RuntimeError(f"Release file mismatch: {name}")
    return sha(manifest)


def configure(bundle, extra=()):
    install_frozen_imports([bundle / "github", *extra])
    directories = [bundle / "github/experiments/unified_reconstruction_20261113"]
    if bundle == CALIBRATED:
        directories.insert(0, bundle / "github/experiments/q217_calibrated_20261114")
    sys.path[:0] = [str(p) for p in directories]
    rt = importlib.import_module("runtime")
    # New explicit run inputs may be in 02/03; reject traversal before resolving them.
    rt.contained = read_contained
    return rt
