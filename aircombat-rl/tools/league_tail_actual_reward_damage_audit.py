"""Separate rear WEZ occupancy from actual health damage after track lock."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from tools import league_thread_benchmark as io
from tools.league_matches import MatchEnv
from experiments.league.tail_reward import geometry
from experiments.league.health_rule_v2 import classify


def match(own,foe,band,n):
    episodes=[]
    for k in range(n//2):
        for seat in ('red','blue'):
            env=MatchEnv(own,foe,seat)
            try:
                obs,info=env.reset(seed=band+k);old=geometry(obs)
                hp,ehp=old['own_health'],old['opp_health'];rear_wez=rear_fire=rear_damage=unreturned=0.
                current_head=legacy_head=dealt_total=received_total=0.;streak=longest=0.
                while True:
                    obs,_,term,trunc,info=env.step(env.own_action(obs));g=geometry(obs)
                    own_hp,opp_hp=float(info['own_health']),float(info['opp_health'])
                    dealt=max(0.,ehp-opp_hp);received=max(0.,hp-own_hp)
                    rear_wez+=.05*g['tail_attack']
                    actual_rear=g['tail_attack'] and dealt>0.
                    rear_fire+=.05*actual_rear;rear_damage+=dealt*actual_rear
                    unreturned+=dealt*(actual_rear and received==0.)
                    streak=streak+.05 if actual_rear else 0.;longest=max(longest,streak)
                    current_head+=received*g['head_on'];legacy_head+=received*old['head_on']
                    dealt_total+=dealt;received_total+=received
                    old=g;hp,ehp=own_hp,opp_hp
                    if term or trunc:break
                assert abs(dealt_total-(1-ehp))<1e-9 and abs(received_total-(1-hp))<1e-9
                assert 0<=unreturned<=rear_damage<=dealt_total+1e-9 and rear_fire<=rear_wez+1e-9
                episodes.append(dict(seed=band+k,seat=seat,result=classify(hp,ehp),own_health=hp,opp_health=ehp,
                    duration=float(env._combat.t),rear_wez_seconds=rear_wez,rear_damage_seconds=rear_fire,
                    rear_damage=rear_damage,rear_damage_without_simultaneous_return_fire=unreturned,
                    max_rear_damage_streak=longest,damage_dealt=dealt_total,damage_received=received_total,
                    head_on_damage_current=current_head,head_on_damage_previous_geometry=legacy_head))
            finally:env.close()
    return dict(own=own['id'],foe=foe['id'],episodes=episodes)


def run(source,out):
    if out.exists():raise FileExistsError(out)
    completed=io.read(source/'completion.json');assert completed['status']=='actual_reward_complete'
    old=io.read(source/'plan.json');io.verify(old)
    by_id={s['id']:s for s in old['opponents']};foes=[by_id[k] for k in ('circler','ace','ddqn_s1')]
    roots={f'{arm}_s{seed}':source/f'{arm}_s{seed}' for arm in ('control','tail') for seed in old['seeds']}
    roles={};plan=dict(source_sha256=dict(old['source_sha256']),input_sha256=dict(old['input_sha256']),processes=8,band=115000000,n=8,opponents=foes,
        scope='Predeclared diagnostic: four control/treatment models x2action replicas x3foes x4consumedIC xbothseats=192games. Rear WEZ is eligibility, not actual firing. Actual damage uses same-step post-motion geometry and health decrement, matching Combat.step ordering. No training/reward/physics change; no promotion.')
    for role,root in roots.items():
        file=root/'result.json';r=io.read(file);roles[role]=[rep['spec'] for rep in r['replicas']];plan['input_sha256'][io.relative(file)]=io.sha(file)
        for s in roles[role]:
            for file in list((io.ROOT/s['design']).glob('*.py'))+[io.ROOT/s['weights']]:plan['input_sha256'][io.relative(file)]=io.sha(file)
    plan['roles']=roles
    references={}
    for ref_root in (source,io.ROOT/'runs/league_tail_retention_broad_20261012'):
        for file in (ref_root/'evaluation').glob('match_*.json'):
            b=io.read(file);role=b['role']
            if ref_root!=source:role={'retention_9400':'parent9400','retention_9501':'parent9501'}.get(role)
            if role not in roles or b['foe'] not in {s['id'] for s in foes}:continue
            plan['input_sha256'][io.relative(file)]=io.sha(file)
            for row in b['episodes']:
                key=(role,b['own'],b['foe'],row['seed'],row['seat']);assert key not in references;references[key]=row
    expected_cells={(role,s['id'],f['id'],plan['band']+k,seat) for role,specs in roles.items() for s in specs for f in foes for k in range(plan['n']//2) for seat in ('red','blue')}
    assert references.keys()==expected_cells and len(references)==192
    for file in (Path(__file__).resolve(),io.ROOT/'aircombat_gym/wvr/envs/base.py',io.ROOT/'experiments/league/tail_reward.py'):
        plan['source_sha256'][io.relative(file)]=io.sha(file)
    io.verify(plan)
    if io.resources()['commit_headroom_gib']<4.5:raise MemoryError('Evaluation startup headroom')
    io.write(out/'plan.json',plan);io.write(out/'runtime.json',dict(pid=os.getpid(),resources=io.resources()))
    started=time.perf_counter();records={role:[] for role in roles};observed=set()
    jobs=[(role,s,f) for role,specs in roles.items() for s in specs for f in foes]
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(match,s,f,plan['band'],plan['n']):(i,role) for i,(role,s,f) in enumerate(jobs)}
        for future in as_completed(futures):
            i,role=futures[future];row=future.result()
            for ep in row['episodes']:
                key=(role,row['own'],row['foe'],ep['seed'],ep['seat']);assert key not in observed;observed.add(key)
                ref=references[key]
                assert ep['result']==classify(ep['own_health'],ep['opp_health'])
                assert all(ep[k]==ref[k] for k in ('own_health','opp_health','result','duration'))
                assert ep['rear_wez_seconds']==ref['tail_attack_seconds']
            io.write(out/f'match_{i:04d}.json',dict(role=role,**row));records[role].extend(row['episodes'])
    expected=sum(len(s) for s in roles.values())*len(foes)*plan['n'];assert expected==192
    assert sum(len(rs) for rs in records.values())==expected and observed==expected_cells
    metrics=('rear_wez_seconds','rear_damage_seconds','rear_damage','rear_damage_without_simultaneous_return_fire','max_rear_damage_streak','damage_dealt','damage_received','head_on_damage_current','head_on_damage_previous_geometry')
    profiles={role:dict(games=len(rs),win_rate=sum(r['result']=='win' for r in rs)/len(rs),rear_damage_games=sum(r['rear_damage']>0 for r in rs),
        zero_received_rear_damage_games=sum(r['rear_damage']>0 and r['damage_received']==0 for r in rs),**{k:sum(r[k] for r in rs)/len(rs) for k in metrics}) for role,rs in records.items()}
    io.verify(plan);report=dict(status='rear_actual_damage_audit_complete',new_training_steps=0,evaluation_games=192,evaluation_seconds=time.perf_counter()-started,profiles=profiles,policy_promoted=False)
    io.write(out/'raw_audit.json',dict(status='192exact_cells_and_original_outcomes_replayed',same_health_verdict_duration_and_rear_wez=True))
    io.write(out/'completion.json',report);print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source.resolve(),a.out.resolve())
