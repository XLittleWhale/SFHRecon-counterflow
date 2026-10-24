"""Export source or research data separately; no network or deletion operations."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
TOP_FILES={'README.md','run.py','pyproject.toml','.gitignore','.gitattributes'}
TOP_DIRS={'src','tests','scripts','docs','config','artifacts','.github'}
SKIP={'__pycache__','.pytest_cache','.git'}

def source_files():
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(ROOT)
        if any(s in SKIP or s.endswith('.egg-info') for s in rel.parts):continue
        if p.suffix in ('.pyc','.log','.tmp'):continue
        if rel.parts[0] in TOP_DIRS or rel.as_posix() in TOP_FILES:
            if p.stat().st_size>50*1024**2:raise ValueError(f'Unexpected large source file: {rel}')
            yield p

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['source','data'])
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    if args.kind=='source':files=list(source_files())
    else:
        files=[p for d in (ROOT/'data',ROOT/'runs/baseline') for p in d.rglob('*') if p.is_file() and not p.is_symlink()]
    if any(out==p.resolve() for p in files):raise ValueError('Output cannot be an input')
    if not files:raise ValueError('No files to package')
    out.parent.mkdir(parents=True,exist_ok=True);manifest={}
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in files:
            rel=p.relative_to(ROOT).as_posix();h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
            manifest[rel]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
            z.write(p,rel)
        z.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2))
    print(json.dumps(dict(kind=args.kind,files=len(files),bytes=out.stat().st_size,output=str(out))))

if __name__=='__main__':main()
