"""One public entry point; no historical version selector."""
import argparse
import importlib
import os
import sys

def main():
    parser=argparse.ArgumentParser(description='Unified probabilistic two-fluid reconstruction')
    parser.add_argument('--data',help='Directory containing the portable data bundle')
    parser.add_argument('--run',help='Separate run/output directory')
    parser.add_argument('command',choices=['verify','train','calibrate','invert','evaluate','plot','steady','particles'])
    args,rest=parser.parse_known_args()
    if args.data:os.environ['COUNTERFLOW_DATA']=args.data
    if args.run:os.environ['COUNTERFLOW_RUN']=args.run
    if args.command in ('steady','particles'):
        module=importlib.import_module('.forward.workflow',__package__)
        sys.argv=[f'counterflow {args.command}']+rest
        return getattr(module,args.command+'_main')()
    module=importlib.import_module('.'+args.command,__package__)
    if args.command=='invert':
        p=argparse.ArgumentParser(prog='counterflow invert')
        p.add_argument('--case',nargs='+',choices=module.CASES,default=module.CASES)
        return module.main(p.parse_args(rest))
    if rest:parser.error('Unrecognized arguments: '+' '.join(rest))
    return module.main()
