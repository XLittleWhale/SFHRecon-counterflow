"""Descriptive optimization curves after fixed-budget experiment completion."""
import json
import numpy as np
import experiment as E
from evaluate_joint import prepare_plotting,save_figure,COLORS,ARM_LABELS

def main():
    E.verify_protocol()
    if not (E.OUT/'selection_frozen.json').exists():raise RuntimeError('Require global selection freeze')
    plt=prepare_plotting()
    fig,axes=plt.subplots(3,2,figsize=(12,11),layout='constrained')
    histories={}
    for column,U in enumerate(E.CASES):
        for arm in E.ARMS:
            runs=[]
            for seed in E.SEEDS:
                folder=E.OUT/'runs'/E.case_name(U)/f'seed_{seed}'/arm
                history=json.loads((folder/'history.json').read_text())
                assert history[-1]['iteration']==E.STEPS
                runs.append(history);histories[(U,seed,arm)]=history
            x=np.arange(1,E.STEPS+1)
            y=np.array([[r['objective'] for r in h[1:]] for h in runs])
            axes[0,column].plot(x,y.mean(0),color=COLORS[arm],label=ARM_LABELS[arm])
            axes[0,column].fill_between(x,y.min(0),y.max(0),color=COLORS[arm],alpha=.15)
            for row,kind in ((1,'free'),(2,'cap')):
                xx=[r['iteration'] for r in runs[0] if 'validation' in r]
                yy=np.array([[r['validation'][kind] for r in h if 'validation' in r] for h in runs])
                axes[row,column].plot(xx,yy.mean(0),color=COLORS[arm],label=ARM_LABELS[arm])
                axes[row,column].fill_between(xx,yy.min(0),yy.max(0),color=COLORS[arm],alpha=.15)
                for h in runs:
                    selected=[r for r in h if r.get('selected')][-1]
                    axes[row,column].scatter(selected['iteration'],selected['validation'][kind],
                        marker='x',color=COLORS[arm],s=35,zorder=5)
        for row,label in enumerate(('0.08 x I NLL + regularization (pre-update)','Validation free NLL','Validation captured NLL')):
            ax=axes[row,column];ax.set(title=f'{U*1000:g} mm/s | {label}',xlabel='Field update')
            ax.axvline(40,color='gray',linestyle=':',linewidth=1);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Optimization trajectories | line: mean; band: initialization range\n'
                 "Vertical line: warm-up end; crosses: each arm's selected field/network checkpoint")
    save_figure(fig,E.OUT/'figures/03_optimization',plt)
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for column,U in enumerate(E.CASES):
        runs=[[r for r in histories[(U,seed,'new_joint')] if 'fine' in r] for seed in E.SEEDS]
        x=[r['iteration'] for r in runs[0]]
        for row,(metric,label) in enumerate((('F_captured_nll','Fine-tuning set captured NLL'),
                ('teacher_student_KL','Teacher-to-student latent KL'),
                ('replay_supervised_loss','Old-case replay loss'))):
            y=np.array([[r['fine'][metric] for r in h] for h in runs])
            ax=axes[row,column];ax.plot(x,y.mean(0),color=COLORS['new_joint'])
            ax.fill_between(x,y.min(0),y.max(0),color=COLORS['new_joint'],alpha=.2)
            ax.set(title=f'{U*1000:g} mm/s | {label}',xlabel='Field update');ax.grid(alpha=.2)
    fig.suptitle('Restricted probability-head adaptation | values before each network update\n'
                 'Continuous-density replay NLL can be negative')
    save_figure(fig,E.OUT/'figures/04_adaptation',plt)
    print('Training figures exported',flush=True)

if __name__=='__main__':main()
