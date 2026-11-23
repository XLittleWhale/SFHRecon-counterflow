"""Collect and later remove old junctions and extra names for shared files."""
import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from .paths import ROOT, REPO, PLAN, SECTIONS, key

PROCESSED = SECTIONS[3] / "inputs/data/experimental"
PROC_TARGET = SECTIONS[5] / "processed/trajectory_inputs"
MODELS = SECTIONS[4] / "results/periodic_baseline/probability_selection"
MODELS_TARGET = SECTIONS[3] / "trained_models/periodic_baseline"


def scan():
    dirs = [ROOT]
    junctions, files = {}, defaultdict(list)
    while dirs:
        base = dirs.pop()
        for entry in os.scandir(base):
            p = Path(entry.path)
            if p.name in (".git", ".venv"):
                continue
            s = entry.stat(follow_symlinks=False)
            if getattr(s, "st_file_attributes", 0) & 1024 or entry.is_symlink():
                target = p.resolve(strict=True)
                if not target.is_relative_to(ROOT) or not target.is_dir():
                    raise ValueError(f"Unexpected project link: {p} -> {target}")
                junctions[p] = target
                continue
            if entry.is_dir(follow_symlinks=False):
                dirs.append(p)
            else:
                # Extended Windows syntax also covers archive members beyond MAX_PATH.
                raw = "\\\\?\\" + str(p) if os.name == "nt" else str(p)
                s = os.stat(raw, follow_symlinks=False)
                if s.st_nlink > 1:
                    files[s.st_dev, s.st_ino].append(p)
    return junctions, files


def rel(p):
    return p.relative_to(ROOT).as_posix()


def preferred(paths):
    def score(p):
        if p.is_relative_to(REPO):
            return 0
        if p.is_relative_to(SECTIONS[2] / "source"):
            return 1
        if p.is_relative_to(SECTIONS[4] / "packages"):
            return 2
        if p.is_relative_to(SECTIONS[6] / "history"):
            return 3
        return 4
    return min(paths, key=lambda p: (score(p), rel(p)))


def plan():
    junctions, files = scan()
    routes = {rel(p): rel(target) for p, target in junctions.items()}
    groups = []
    for paths in files.values():
        if len(paths) < 2:
            continue
        keep = preferred(paths)
        for old in sorted(set(paths) - {keep}):
            routes[rel(old)] = rel(keep)
            groups.append((rel(old), rel(keep)))
    moves = [(PROCESSED, PROC_TARGET), (MODELS, MODELS_TARGET)]
    for src, dst in moves:
        if junctions.get(dst) != src:
            raise RuntimeError(f"Unexpected original for {dst}")
        routes.pop(rel(dst))
        routes[rel(src)] = rel(dst)
    # Frozen Python modules refer both to individual artifacts and the artifact directory.
    directory = SECTIONS[3] / "dependencies/frozen_assets/artifacts"
    if directory.is_dir():
        routes[rel(directory)] = rel(REPO / "artifacts")
    # The old periodic publication result no longer contains the selected models copy.
    old = ROOT / "VFM/2DCounterflow/outputs/PINNTwoFluid_20261105/probability_selection"
    routes[rel(old)] = rel(MODELS_TARGET)
    payload = dict(original_root=str(ROOT), routes=dict(sorted(routes.items())),
                   junctions=[(rel(p), rel(q)) for p,q in sorted(junctions.items())],
                   redundant_link_names=groups, moves=[(rel(p),rel(q)) for p,q in moves])
    PLAN.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Path map: {len(junctions)} junctions, {len(groups)} redundant file names, {len(moves)} physical moves.")


def apply():
    data = json.loads(PLAN.read_text(encoding="utf-8"))
    links, dupes = dict(data["junctions"]), dict(data["redundant_link_names"])
    junctions, files = scan()
    assert {rel(p): rel(q) for p,q in junctions.items()} == links
    assert sum(max(len(x)-1, 0) for x in files.values()) == len(dupes)
    for raw, keep in dupes.items():
        p,q = ROOT/raw,ROOT/keep
        assert p.resolve() == p and q.resolve() == q and p.samefile(q) and p.is_file()
    protected = set(links) | set(dupes)
    assert all(not {".git",".venv"}.intersection(Path(p).parts) for p in protected)
    for raw, target in data["junctions"]:
        p,q = ROOT/raw,ROOT/target
        assert p.parent.resolve() == p.parent and p.resolve() == q
    for raw, _ in data["junctions"]:
        p = ROOT/raw
        assert p.parent.resolve() == p.parent and (getattr(p.lstat(),"st_file_attributes",0)&1024)
        p.rmdir()  # On Windows rmdir of the verified junction removes only the link.
    for src,dst in data["moves"]:
        p,q=ROOT/src,ROOT/dst
        assert p.is_relative_to(ROOT) and q.is_relative_to(ROOT)
        assert p.resolve() == p and q.parent.resolve() == q.parent and not q.exists()
        p.rename(q)
    for raw, keep in dupes.items():
        p,q=ROOT/raw,ROOT/keep
        assert p.samefile(q)
        p.unlink()
    cleanup = {p for raw in protected for p in (ROOT/raw).parents if p.is_relative_to(ROOT)}
    for p in sorted(cleanup, key=lambda t: -len(t.parts)):
        if p in (*SECTIONS.values(), ROOT, REPO) or p.name in ("source","dependencies","history","temp","results"):
            continue
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    after, remaining=scan()
    assert not after and not any(len(x)>1 for x in remaining.values())
    print(f"Removed {len(links)} junctions and {len(dupes)} shared-file aliases; originals retained.")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--apply",action="store_true")
    apply() if p.parse_args().apply else plan()
