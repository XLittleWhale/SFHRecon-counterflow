"""Compact read-only progress view; safe to call while workers update checkpoints."""
from common import *

if __name__=='__main__':
    path=OUT/'pipeline_status.json'
    if not path.exists():
        print('Pipeline has not started.')
    else:
        p=read(path)
        print('Stage:',p['stage'],'| complete:',p['complete'],'| updated UTC:',p['updated_utc'])
        print('Finished:',len(p['finished']),'/',p['total_runs'],'| probability updates: 0')
        if p.get('active'):
            row=p['active'];field=row.get('field',{})
            print(row['case'],row['seed'],'PID',row['pid'],
                '| field step',field.get('step','-'),'/',max(BUDGETS))
        if p.get('error'): print('Failure:',p['error'])
