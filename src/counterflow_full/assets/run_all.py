"""Launch from any cwd. No automatic CUDA -> long CPU fallback."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=root.parent/'fullwindow_4mm_results')
    p.add_argument('--phase',choices=['all','prepare','train','calibrate','tune','invert','evaluate','check'],default='all')
    p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    phases=['prepare','train','calibrate','tune','invert','evaluate'] if a.phase=='all' else [a.phase]
    for phase in phases:
        cmd=[sys.executable,'-m','counterflow_full.run','--package',str(root),'--out',str(a.out.resolve()),'--phase',phase,'--device',a.device]
        if a.smoke:cmd.append('--smoke')
        print('Starting',phase,flush=True);subprocess.run(cmd,cwd=root,check=True)
    print('Outputs:',a.out.resolve(),flush=True)


if __name__=='__main__':main()
