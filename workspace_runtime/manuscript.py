"""Inspect or verify the external assets for the Git-versioned manuscript baseline."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "config/manuscript_baseline.json"


def verify(root, baseline):
    root = root.resolve(strict=True)
    failures = []
    for name, record in baseline["workspace_files"].items():
        relative = Path(name)
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError(f"Invalid workspace path in manifest: {name}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(f"{name}: unavailable")
            continue
        with path.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != record["sha256"]:
            failures.append(f"{name}: hash mismatch")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--list", action="store_true", help="show cases without reading workspace files")
    args = parser.parse_args(argv)
    baseline = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.list:
        print(json.dumps({"id": baseline["id"], "cases": baseline["cases"]}, indent=2))
        return 0
    root = args.workspace or next((p for p in REPO.parents if (p/"01_Documentation").is_dir()), None)
    if root is None:
        parser.error("provide --workspace <SFHFlow> to verify the data and result files")
    errors = verify(root, baseline)
    for line in errors:
        print(line)
    if errors:
        return 1
    print(f'{baseline["id"]}: verified {len(baseline["workspace_files"])} workspace files')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
