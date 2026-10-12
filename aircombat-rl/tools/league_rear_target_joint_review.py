"""Verify paired joint search and evaluate only genuinely new finalists."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import time
import numpy as np
from tools import league_thread_benchmark as io
from tools.league_tail_actual_reward_damage_audit import match
from tools.league_tail_broad_teacher_search import fitness
from experiments.league.health_rule_v2 import classify

def profile(rows):
    return dict(games=len(rows),wins=sum(e['result']=='win' for e in rows),rear_damage_games=sum(e['rear_damage']>0 for e in rows),zero_hit_winning_rear_games=sum(e['result']=='win' and e['rear_damage']>0 and e['damage_received']==0 for e in rows),**{k:sum(e[k] for e in rows)/len(rows) for k in ('rear_damage','rear_wez_seconds','damage_dealt','damage_received','head_on_damage_current')})

def run():
    source=io.ROOT/'runs/league_rear_target_joint_20261012';out=io.ROOT/'runs/league_rear_target_joint_review_20261012'
    if out.exists():raise FileExistsError(out)
    plan=io.read(source/'plan.json');io.verify(plan);done=io.read(source/'completion.json');assert done['status']=='rear_target_joint_complete'
    cells=set();steps=0;best={};anchors={};audit_started=time.perf_counter()
    for arm in ('control','aft'):
        for g in range(8):
            folder=source/arm/f'g{g}';pop=io.read(folder/'population.json');result=io.read(folder/'result.json')
            for f,h in pop['artifact_sha256'].items():assert io.sha(io.ROOT/f)==h
            if arm=='control':assert all(p[-1]==0 for p in pop['parameters'])
            for i,s in enumerate(pop['specs']):
                rows=[]
                for foe in plan['foes']:
                    b=io.read(folder/'matches'/(s['id']+'_'+foe['id']+'.json'))
                    assert b['own']==s['id'] and b['foe']==foe['id']
                    if i<3:
                        key=(i,foe['id'])
                        if key in anchors:assert b['episodes']==anchors[key]
                        else:anchors[key]=b['episodes']
                    for e in b['episodes']:
                        key=(arm,g,s['id'],foe['id'],e['seed'],e['seat']);assert key not in cells;cells.add(key)
                        assert e['result']==classify(e['own_health'],e['opp_health'])
                        assert abs(e['own_health']+e['damage_received']-1)<1e-8
                        steps+=round(e['duration']/.05);rows.append(dict(e,foe=foe['id']))
                rows.sort(key=lambda r:(r['foe'],r['seed'],r['seat']))
                rank=next(r for r in result['ranking'] if r['spec']['id']==s['id'])
                assert rank['parameters']==pop['parameters'][i]
                assert abs(fitness(rows)['score']-rank['score'])<1e-12
            r=result['ranking'][0]
            if arm not in best or r['score']>best[arm]['score']:best[arm]=r
        assert best[arm]==done['arms'][arm]['best']
    c=io.read(source/'control/g0/population.json')['parameters'];a=io.read(source/'aft/g0/population.json')['parameters'];assert np.array_equal(np.array(c)[:,:17],np.array(a)[:,:17])
    expected={(arm,g,f'{arm}_g{g}_c{i}',f['id'],441000010+k,seat) for arm in ('control','aft') for g in range(8) for i in range(16) for f in plan['foes'] for k in (0,1) for seat in ('red','blue')}
    assert cells==expected and len(cells)==6144 and steps==done['simulation_steps']
    io.write(source/'raw_audit.json',dict(status='6144_exact_cells_scores_steps_and_common_anchors_verified',audit_seconds=time.perf_counter()-audit_started))
    base=io.ROOT/'runs/league_selector_experiment_20261012';old=io.read(base/'plan.json');io.verify(old)
    records=[];inputs=dict(old['input_sha256']);expert_records={}
    for f in (base/'evaluation').glob('match_*.json'):
        b=io.read(f)
        if b['role'] in ('expert0','expert1','expert2'):
            records.append(b);expert_records.setdefault(b['role'],[]).append(b);inputs[io.relative(f)]=io.sha(f)
    assert sum(len(b['episodes']) for b in records)==288
    rear=io.read(io.ROOT/'runs/league_tail_teacher_search_20261012/completion.json')['best']['spec']['parameters']
    lag=io.read(io.ROOT/'runs/league_tail_broad_teacher_search_20261012/completion.json')['best']['spec']['parameters']
    new={};reuse={}
    for arm,r in best.items():
        pars=r['parameters']
        if pars[-1]==0 and pars[:-1]==rear:
            for b in expert_records['expert2']:records.append(dict(b,role=arm,own=r['spec']['id']))
            reuse[arm]='old_rear_teacher_exact_parameters_and_zero_offset'
        elif pars[-1]==0 and pars[:-1]==lag:
            prev=io.ROOT/'runs/league_tail_broad_teacher_confirm_20261012';io.verify(io.read(prev/'plan.json'))
            for f in prev.glob('*.json'):
                b=io.read(f)
                if b.get('role')=='candidate':records.append(dict(b,role=arm,own=r['spec']['id']));inputs[io.relative(f)]=io.sha(f)
            reuse[arm]='novel_lag_fast_exact_parameters_and_zero_offset'
        else:new[arm]=r['spec']
    for f in [source/'completion.json',source/'raw_audit.json']:inputs[io.relative(f)]=io.sha(f)
    evalplan=dict(source_sha256=dict(plan['source_sha256']),input_sha256=inputs,roles={k:r['spec'] for k,r in best.items()},new_roles=new,reuse=reuse,foes=old['development_opponents'],band=115000000,n=4,processes=8,scope='Consumed development conditions; 96 cells per candidate plus288 frozen expert cells. Only newly parameterized candidates simulated. No final test/no automatic promotion.')
    evalplan['source_sha256'][io.relative(Path(__file__))]=io.sha(Path(__file__))
    for spec in evalplan['roles'].values():
        for f in [io.ROOT/spec['weights'],*(io.ROOT/spec['design']).glob('*.py')]:evalplan['input_sha256'][io.relative(f)]=io.sha(f)
    io.verify(evalplan)
    if io.resources()['commit_headroom_gib']<4.5:raise MemoryError('Evaluation startup headroom')
    io.write(out/'plan.json',evalplan);io.write(out/'runtime.json',dict(pid=os.getpid(),resources=io.resources()))
    t=time.perf_counter()
    if new:
        with ProcessPoolExecutor(max_workers=8) as pool:
            jobs={pool.submit(match,s,f,115000000,4):(role,f['id']) for role,s in new.items() for f in evalplan['foes']}
            for future in as_completed(jobs):
                role,foe=jobs[future];b=dict(role=role,**future.result());records.append(b);io.write(out/'matches'/f'{role}_{foe}.json',b)
    seconds=time.perf_counter()-t if new else 0.
    cells=set()
    for b in records:
        for e in b['episodes']:
            key=(b['role'],b['foe'],e['seed'],e['seat']);assert key not in cells;cells.add(key)
            assert e['result']==classify(e['own_health'],e['opp_health'])
    expected={(r,f['id'],115000000+k,seat) for r in ('control','aft','expert0','expert1','expert2') for f in evalplan['foes'] for k in (0,1) for seat in ('red','blue')}
    assert cells==expected and len(cells)==480
    profiles={}
    for role in ('control','aft','expert0','expert1','expert2'):
        profiles[role]={}
        for foe in ['all','noncircler']+[f['id'] for f in evalplan['foes']]:
            rows=[e for b in records if b['role']==role and (foe=='all' or (foe=='noncircler' and b['foe']!='circler') or b['foe']==foe) for e in b['episodes']]
            profiles[role][foe]=profile(rows)
    io.verify(evalplan)
    report=dict(status='rear_target_joint_review_complete',new_evaluation_games=96*len(new),reused_evaluation_games=480-96*len(new),evaluation_seconds=seconds,profiles=profiles,reuse=reuse,policy_promoted=False)
    io.write(out/'completion.json',report);print({r:p['all'] for r,p in profiles.items()},flush=True)

if __name__=='__main__':run()
