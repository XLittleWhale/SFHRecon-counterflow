"""Explicit path migration for old, SHA-bound records; new runs use physical paths."""
import importlib.machinery
import importlib.util
import ast
import json
import os
import re
import sys
from pathlib import Path as PhysicalPath, PurePath

ROOT = next(p for p in PhysicalPath(__file__).resolve().parents if (p / "01_Documentation").is_dir())
REPO = PhysicalPath(__file__).resolve().parents[1]
BASE = REPO.parent
SECTIONS = {int(p.name[:2]): p for p in ROOT.iterdir() if p.name[:2] in
            ("01", "02", "03", "04", "05", "06")}
PLAN = REPO / "workspace_runtime" / "legacy_paths.json"


def key(path):
    return os.path.normcase(os.path.normpath(os.fspath(path)))


_routes = {}
_ordered = ()


def add_routes(rows):
    global _ordered
    for old, new in rows:
        old, new = key(old), PhysicalPath(new)
        if not new.is_absolute() or not PurePath(new).is_relative_to(ROOT):
            raise ValueError(f"Path route outside the project: {new}")
        if old != key(new):
            _routes[old] = str(new)
    _ordered = tuple(sorted(_routes, key=len, reverse=True))


def physical(path):
    current = os.path.normpath(os.fspath(path))
    seen = set()
    while True:
        normalized = key(current)
        if normalized in seen:
            raise RuntimeError(f"Cyclic migration route: {path}")
        seen.add(normalized)
        for old in _ordered:
            if normalized == old or normalized.startswith(old + os.sep):
                current = _routes[old] + current[len(old):]
                break
        else:
            return PhysicalPath(current)


class LegacyPath(type(PhysicalPath())):
    """Path syntax for frozen modules; logical path parts remain available for manifests."""
    def __str__(self):
        return str(physical(super().__str__()))


def load_map():
    data = json.loads(PLAN.read_text(encoding="utf-8"))
    old_root = PhysicalPath(data["original_root"])
    pairs = []
    for old, new in data["routes"].items():
        for root in dict.fromkeys((ROOT, old_root)):
            pairs.append((root / old, ROOT / new))
    add_routes(pairs)


_PATH_IMPORT = re.compile(r"(?m)^from pathlib import Path[ \t]*$")
_IO = {("torch","load"), ("np","load"), ("numpy","load")}


def frozen_input(value):
    if isinstance(value, (str, os.PathLike)):
        return physical(value)
    return value


class FrozenIO(ast.NodeTransformer):
    """Cover old records passed straight as strings to file-loading functions."""
    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        mapped = isinstance(f,ast.Name) and f.id == "open"
        if isinstance(f,ast.Attribute) and isinstance(f.value,ast.Name):
            mapped |= (f.value.id,f.attr) in _IO
        if mapped and node.args and not isinstance(node.args[0],ast.Starred):
            old = node.args[0]
            node.args[0] = ast.copy_location(ast.Call(
                func=ast.Name(id="_sfhflow_frozen_input",ctx=ast.Load()),args=[old],keywords=[]),old)
        return node


def install_frozen_imports(roots):
    """Resolve paths in selected frozen modules without rewriting their original bytes."""
    roots = tuple(PhysicalPath(p).resolve() for p in roots)
    loader = importlib.machinery.SourceFileLoader
    old_get_code = loader.get_code
    sys.modules["sfhflow_frozen_paths"] = sys.modules[__name__]

    def get_code(self, name):
        file = PhysicalPath(self.path).resolve()
        if not any(file.is_relative_to(root) for root in roots):
            return old_get_code(self, name)
        source = importlib.util.decode_source(self.get_data(self.path))
        source = _PATH_IMPORT.sub("from sfhflow_frozen_paths import LegacyPath as Path", source)
        tree = ast.parse(source, self.path)
        tree = FrozenIO().visit(tree)
        header = ast.parse("from sfhflow_frozen_paths import frozen_input as _sfhflow_frozen_input").body[0]
        i = int(bool(tree.body and isinstance(tree.body[0],ast.Expr)
                     and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str)))
        while i < len(tree.body) and isinstance(tree.body[i],ast.ImportFrom) and tree.body[i].module=="__future__":
            i += 1
        tree.body.insert(i,header)
        return compile(ast.fix_missing_locations(tree), self.path, "exec", dont_inherit=True)

    loader.get_code = get_code


def read_contained(root, name):
    """Validate a logical archive/run path before mapping authorized historical inputs."""
    name = PurePath(name)
    if name.is_absolute() or name.drive or ".." in name.parts:
        raise ValueError(f"Path escapes the run: {name}")
    return LegacyPath(root) / name
