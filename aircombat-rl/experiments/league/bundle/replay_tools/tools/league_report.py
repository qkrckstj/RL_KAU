"""Read-only league summaries and static cross-play figures (base Python + matplotlib)."""
from argparse import ArgumentParser
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def report(run,out,plot=False):
    out.mkdir(parents=True,exist_ok=True)
    result={}
    baseline=read(run/'baseline/results.json')
    result['baseline']=[{k:r[k] for k in ('own','foe','band','summary')} for r in baseline]
    result['baseline_scope']='Six shared ICs, both seats, 12 games per pairing; development only'
    main=run/'main'
    if (main/'history.json').exists():
        history=read(main/'history.json')
        result['rounds']=[dict(round=r['round'],challenger=r['challenger']['id'],
            champion=r['champion']['id'],opponents=len(r['pool_ids']),admission=r['admission'],
            training_games=r['search']['training_games'],training_steps=r['search']['training_steps']) for r in history]
    if (main/'progress.json').exists():
        result['last_recorded_progress']=read(main/'progress.json')
    followup=run/('followup_pool' if (run/'followup_pool').exists() else 'followup')
    result['followup_directory']=str(followup)
    if (followup/'progress.json').exists():
        result['last_recorded_followup_progress']=read(followup/'progress.json')
    if (followup/'verdict.json').exists():
        result['final_evaluation']=read(followup/'verdict.json')
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    if not plot:
        print(json.dumps(result.get('rounds',result.get('last_recorded_progress')),ensure_ascii=False))
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    ids=[]
    for row in baseline:
        for identity in (row['own'],row['foe']):
            if identity not in ids:
                ids.append(identity)
    values=np.full((len(ids),len(ids)),np.nan)
    draws=values.copy()
    for row in baseline:
        i,j=ids.index(row['own']),ids.index(row['foe'])
        s=row['summary']
        values[i,j]=s['wins']/s['n']
        values[j,i]=s['losses']/s['n']
        draws[i,j]=draws[j,i]=s['draws']/s['n']
    labels=['CEM original','DDQN seed 1','Ace','Pursuit','Evader','Fixed left','Fixed right']
    fig,axes=plt.subplots(1,2,figsize=(13.5,6),layout='constrained')
    for ax,data,title in zip(axes,(values,draws),('Win rate','Draw rate (mutual or timeout)')):
        ax.imshow(data,vmin=0,vmax=1,cmap='Blues')
        ax.set(xticks=range(len(ids)),yticks=range(len(ids)),xticklabels=labels,yticklabels=labels,
               title=title,xlabel='Opponent',ylabel='Policy')
        plt.setp(ax.get_xticklabels(),rotation=40,ha='right')
        for i in range(len(ids)):
            for j in range(len(ids)):
                value=data[i,j]
                ax.text(j,i,'-' if np.isnan(value) else f'{value:.0%}',ha='center',va='center',
                        color='white' if value>.6 else '#182530',fontsize=10)
    fig.suptitle('Initial policy cross-play | Official FairFight\n6 shared initial conditions x both seats = 12 games per pair (development)',fontsize=13)
    fig.savefig(out/'baseline_crossplay.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/league_20261006')
    parser.add_argument('--out',type=Path,default=ROOT/'experiments/league/results')
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    report(args.run,args.out,args.plot)
