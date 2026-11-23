"""Compact read-only progress view; safe to call while workers update checkpoints."""
from common import *

if __name__=='__main__':
    path=OUT/'pipeline_status.json'
    if not path.exists():
        print('Pipeline has not started.')
    else:
        p=read(path)
        print('Stage:',p['stage'],'| complete:',p['complete'],'| updated UTC:',p['updated_utc'])
        print('Finished:',len(p['finished']),'/ 9 | active:',len(p['active']),'| pending:',len(p['pending']))
        for row in p['active']:
            prob,field=row.get('probability',{}),row.get('field',{})
            print(row['case'],row['seed'],'PID',row['pid'],
                '|',row.get('job',{}).get('stage','starting'),
                '| probability epoch',prob.get('epoch','-'),'/ 80 maximum',
                '| field step',field.get('step','-'),'/ 600')
        if p['failed']: print('Failures:',p['failed'])
