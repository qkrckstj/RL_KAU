"""Bounded CEM search for diverse rear-damage demonstrations; original policies frozen."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from tools import league_thread_benchmark as io
from tools.league_tail_actual_reward_damage_audit import match
from experiments.league.controller import LOW,HIGH,embed,Policy


def fitness(rows):
    win=np.mean([r['result']=='win' for r in rows]);rear=np.mean([r['rear_damage'] for r in rows]);received=np.mean([r['damage_received'] for r in rows])
    # A minimum across named training opponents discourages one-foe specialization.
    perfoe={f:np.mean([r['rear_damage'] for r in rows if r['foe']==f]) for f in sorted({r['foe'] for r in rows})}
    value=.5*win+rear+.5*min(perfoe.values())-.25*received
    return dict(score=float(value),win_rate=float(win),rear_damage=float(rear),damage_received=float(received),rear_by_foe={k:float(v) for k,v in perfoe.items()})


def run(out):
    if out.exists():raise FileExistsError(out)
    source=io.ROOT/'runs/league_tail_actual_reward_20261012';old=io.read(source/'plan.json');io.verify(old)
    original=next(s for s in old['opponents'] if s['id']=='cem_original')
    anchors=[embed(original['parameters'])]+[np.array(s['parameters']) for s in old['opponents'] if s['kind']=='reactive']
    for a in anchors:Policy(parameters=a)
    by_id={s['id']:s for s in old['opponents']}
    foes=[by_id[k] for k in ('ace','lead','pursuit','circler','extend_cem_s6800_g3_c9','episode_mix_sampled_4900')]
    plan=dict(source_sha256=dict(old['source_sha256']),input_sha256=dict(old['input_sha256']),generations=4,population=16,elite=4,processes=8,band=441000010,n=4,foes=foes,anchors=[a.tolist() for a in anchors],seed=10201,
              scope='1536 parameter-search games:4generations16candidates6foes2trainingIC both seats. Only roster expanded; objective coefficients and population budget unchanged. Twice the old match budget, not equal-compute algorithm evidence. No PPO updates. Raw observations only; no opponent ID input. Search fitness .5win+rear+.5worst-foe-rear-.25received. Consumed training range, not held-out validation. Original CEM/PPO unchanged.')
    for f in (Path(__file__),io.ROOT/'tools/league_tail_actual_reward_damage_audit.py',io.ROOT/'experiments/league/controller.py'):plan['source_sha256'][io.relative(f)]=io.sha(f)
    io.verify(plan)
    if io.resources()['commit_headroom_gib']<4.5:raise MemoryError('Startup headroom')
    io.write(out/'plan.json',plan);io.write(out/'runtime.json',dict(pid=os.getpid(),resources=io.resources()))
    rng=np.random.default_rng(plan['seed']);mean=anchors[0].copy();std=(HIGH-LOW)*.12;best=None;allrank=[];started=time.perf_counter();games=0;steps=0
    with ProcessPoolExecutor(max_workers=8) as pool:
        for generation in range(4):
            if io.resources()['commit_headroom_gib']<2.8:raise MemoryError('Between-generation headroom')
            candidates=[np.clip(rng.normal(mean,std),LOW,HIGH) for _ in range(16)]
            for i,a in enumerate(anchors):candidates[i]=a.copy()
            if best is not None:candidates[3]=np.array(best['spec']['parameters'])
            specs=[dict(id=f'broad_teacher_g{generation}_c{i}',kind='reactive',parameters=a.tolist()) for i,a in enumerate(candidates)]
            for s in specs:Policy(parameters=s['parameters'])
            rows={s['id']:[] for s in specs}
            jobs={pool.submit(match,s,f,plan['band'],plan['n']):(s,f) for s in specs for f in foes}
            for future in as_completed(jobs):
                s,f=jobs[future];b=future.result();io.write(out/f'g{generation}'/(s['id']+'_'+f['id']+'.json'),b)
                for r in b['episodes']:
                    rows[s['id']].append(dict(r,foe=f['id']));steps+=round(r['duration']/.05)
                games+=len(b['episodes'])
            ranked=[dict(spec=s,**fitness(rows[s['id']])) for s in specs]
            ranked.sort(key=lambda x:x['score'],reverse=True)
            if best is None or ranked[0]['score']>best['score']:best=ranked[0]
            elite=np.array([r['spec']['parameters'] for r in ranked[:4]])
            mean=.3*mean+.7*elite.mean(axis=0);std=np.maximum(.3*std+.7*elite.std(axis=0),(HIGH-LOW)*.025)
            allrank.append(ranked);io.write(out/f'g{generation}'/'ranking.json',ranked);io.verify(plan)
            print('generation',generation,'games',games,'best',{k:v for k,v in best.items() if k!='spec'},flush=True)
    assert games==1536
    io.verify(plan)
    done=dict(status='teacher_search_complete',parameter_search_games=games,simulation_steps=steps,ppo_steps=0,wall_seconds=time.perf_counter()-started,best=best,policy_promoted=False,demonstrations_collected=False)
    io.write(out/'completion.json',done);print(done,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out.resolve())
