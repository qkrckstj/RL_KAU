"""After the active league finishes, diagnose the persistent no-contact runner.

This is open development data. Hand-configured pursuit probes are not described
as newly learned models. Failure of this finite set cannot prove impossibility.
"""
from argparse import ArgumentParser
from pathlib import Path
import time
from experiments.league.controller import embed
from tools.autolab_cem import ROOT,read,write,sha
from tools.league_matches import evaluate_jobs,MatchEnv
from tools.league_train import entrant,verify
from tools.league_validate import process_live


def probes():
    result=[]
    for speed in (600.,650.):
        for lead in (0.,3.,6.):
            p=embed([0.,speed,speed,lead,0.,.3,3.])
            result.append(entrant(p,f'pursuit_{int(speed)}_lead{int(lead)}'))
    for switch in (1000.,1500.):
        p=embed([0.,650.,500.,6.,0.,.3,3.])
        p[9:11]=[switch,650.]
        result.append(entrant(p,f'fast_until_{int(switch)}'))
    return result


def trace(spec,foe,band):
    traces=[]
    for seat in ('red','blue'):
        env=MatchEnv(spec,foe,seat)
        try:
            obs,info=env.reset(seed=band)
            frames=[]
            steps=0
            while True:
                action=env.own_action(obs)
                obs,_,terminal,truncated,info=env.step(action)
                steps+=1
                if steps%5==0 or terminal or truncated:
                    frames.append(dict(t=info['t'],action=int(action),range=info['range'],
                        own_speed=info['own_speed'],opponent_speed=float((sum(float(v)**2 for v in obs[18:21]))**.5/.514444),
                        own_x=float(obs[0]),own_y=float(obs[1]),opp_x=float(obs[15]),opp_y=float(obs[16]),
                        own_health=float(obs[30]),opp_health=float(obs[31])))
                if terminal or truncated:
                    break
            traces.append(dict(model=spec['id'],foe=foe['id'],seat=seat,seed=band,outcome=info['outcome'],frames=frames))
        finally:
            env.close()
    return traces


def run(main,followup,out,pids):
    if out.exists():
        raise FileExistsError('Use a new diagnostic output')
    out.mkdir(parents=True)
    candidates=probes()
    plan=dict(band=33000000,n=12,foe=dict(id='evader',kind='bot',name='evader'),
        candidates=candidates,waiting_for_pids=pids,source_sha256=sha(Path(__file__)),
        parent_rule='Frozen development-selected refinement, otherwise development pool choice, otherwise main champion; never ranked by sealed test',
        scope='Exploratory runner diagnosis, same official physics and 120s, no impossibility claim')
    write(out/'plan.json',plan)
    while True:
        active=[p for p in pids if process_live(p)]
        if not active:
            break
        write(out/'progress.json',dict(stage='waiting_for_existing_experiments',active_pids=active,checked_at=time.time()))
        time.sleep(15)
    verify(read(main/'plan.json'))
    if (followup/'frozen_selection.json').exists():
        parent=read(followup/'frozen_selection.json')['chosen']
    elif (followup/'pool_selection.json').exists():
        parent=read(followup/'pool_selection.json')['chosen']['spec']
    else:
        parent=read(main/'completion.json')['champion']
    write(out/'selected_parent.json',parent)
    write(out/'progress.json',dict(stage='runner_comparison'))
    candidates=[parent]+candidates
    jobs=[dict(own=s,foe=plan['foe'],band=plan['band'],n=plan['n']) for s in candidates]
    results=evaluate_jobs(jobs,out/'matches')
    write(out/'results.json',results)
    best_index=max(range(len(results)),key=lambda i:(results[i]['summary']['wins'],
                   -results[i]['summary']['min_range'],results[i]['summary']['wez_time']))
    best=candidates[best_index]
    records=trace(parent,plan['foe'],plan['band'])
    if best['id']!=parent['id']:
        records+=trace(best,plan['foe'],plan['band'])
    write(out/'traces.json',records)
    improvement=results[best_index]['summary']['wins']>results[0]['summary']['wins']
    write(out/'completion.json',dict(status='complete',parent=results[0]['summary'],best=best,
          best_summary=results[best_index]['summary'],observed_improvement=improvement,
          next='Train and verify a conditional pursuit response, with a new future final test band' if improvement
               else 'No tested pursuit probe improved wins; inspect the recorded approach. This does not prove the fight unwinnable.'))
    write(out/'progress.json',dict(stage='complete'))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--main',type=Path,required=True)
    p.add_argument('--followup',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--wait-pids',nargs='+',type=int,required=True)
    a=p.parse_args()
    try:
        run(a.main.resolve(),a.followup.resolve(),a.out.resolve(),a.wait_pids)
    except Exception as e:
        write(a.out/'failure.json',dict(error=repr(e)))
        raise
