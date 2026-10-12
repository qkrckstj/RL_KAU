"""Geometry-conditioned opponent weights only; same retained parents and frozen controls."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import time,argparse,traceback,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from tools import league_thread_benchmark as io
from tools.league_rear_geometry_train import train
from tools.league_tail_actual_reward_damage_audit import match
from tools.league_rear_target_joint_review import profile
from experiments.league.health_rule_v2 import classify


def run(out,smoke):
    if out.exists():raise FileExistsError(out)
    source=io.ROOT/'runs/league_tail_actual_reward_20261012';old=io.read(source/'plan.json');io.verify(old)
    assert io.read(source/'raw_audit.json')['status']=='exact_cells_verdict_metrics_verified'
    plan=dict(old,source_sha256=dict(old['source_sha256']),input_sha256=dict(old['input_sha256']),opponent_schedule=dict(enabled=True,focus_ids=['fixed_right','league_residual_continuation_20261007_s3301_t2097152'],quarter_focus_ids=['extend_cem_s6800_g3_c9','switching_turn_strong'],fade_steps_per_env=10240),rear_damage_coefficient=.5,position_per_second=.01,development_n=4,chunk_steps=20480 if smoke else old['chunk_steps'],seeds=old['seeds'][:1] if smoke else old['seeds'],training_bands=old['training_bands'][:1] if smoke else old['training_bands'],experiment='geometry_conditioned_opponent_weights',scope='Only geometry-conditioned opponent weighting, fade10240steps/env. Same retained parents/Adam, seeds, rewards, quarter50, full48foe coverage, prior6, gamma.999. Consumed development comparison, not independent final.')
    for f in [Path(__file__),io.ROOT/'tools/league_rear_geometry_train.py',io.ROOT/'experiments/league/rear_geometry_opponents.py',io.ROOT/'tests/test_rear_geometry_opponents.py',io.ROOT/'tools/league_rear_target_joint_review.py',io.ROOT/'tools/league_tail_actual_reward_damage_audit.py']:
        plan['source_sha256'][io.relative(f)]=io.sha(f)
    for f in [source/'plan.json',source/'completion.json',source/'raw_audit.json']:
        plan['input_sha256'][io.relative(f)]=io.sha(f)
    for seed in plan['seeds']:
        f=source/f'tail_s{seed}/result.json';plan['input_sha256'][io.relative(f)]=io.sha(f)
    if not smoke:
        f=io.ROOT/'runs/league_rear_geometry_smoke_20261012/completion.json';assert io.read(f)['status']=='rear_geometry_smoke_complete';plan['input_sha256'][io.relative(f)]=io.sha(f)
    io.verify(plan)
    if io.resources()['commit_headroom_gib']<4.5:raise MemoryError('Startup headroom')
    io.write(out/'plan.json',plan);io.write(out/'runtime.json',dict(pid=os.getpid(),resources=io.resources()))
    started=time.perf_counter();results={};roles={}
    for seed,band in zip(plan['seeds'],plan['training_bands']):
        name=f'geometry_s{seed}';print('learning',name,plan['chunk_steps'],flush=True)
        r=train(plan,plan['warm_parents'][f'control_s{seed}'],seed,band,out/name);results[name]=r
        control=io.read(source/f'tail_s{seed}/result.json')
        assert r['initial_parameters']==control['initial_parameters']
        assert r['actual_learner_seed']==control['actual_learner_seed']==seed
        assert r['worker_gammas']==control['worker_gammas']==[.999]*32
        assert r['worker_curricula']==control['worker_curricula']==[dict(stage='quarter',fraction=.5)]*32
        assert r['worker_rear_coefficients']==control['worker_rear_coefficients']==[.5]*32
        assert plan['position_per_second']==.01
        assert plan['tail_coefficients']==old['tail_coefficients']
        assert r['steps']==plan['chunk_steps'] and (smoke or r['steps']==control['steps'])
        episodes=[json.loads(x) for x in (out/name/'training_episodes.jsonl').read_text('utf-8').splitlines()]
        assert len(episodes)==r['training_episodes'] and episodes
        for e in episodes:
            assert e['result']==classify(e['own_health'],e['opp_health'])
            assert 0<=e['damage_received']<=1.+1e-9
            assert e['opponent_focus_ids']==plan['opponent_schedule']['quarter_focus_ids' if e['curriculum_geometry']=='training_tail_quarter' else 'focus_ids']
            assert abs(e['opponent_focus_weight']-max(0.,1.-e['schedule_start_step']/10240))<1e-12
        roles[name]=[x['spec'] for x in r['replicas']]
        roles[f'baseline_s{seed}']=[x['spec'] for x in control['replicas']]
        io.write(out/'training_results.json',results);io.verify(plan)
    if smoke:
        done=dict(status='rear_geometry_smoke_complete',training_steps=sum(r['steps'] for r in results.values()),training_seconds=sum(r['learning_seconds'] for r in results.values()),training_games=sum(r['training_episodes'] for r in results.values()),evaluation_games=0,total_seconds=time.perf_counter()-started,policy_promoted=False)
        io.write(out/'completion.json',done);print(done,flush=True);return
    for specs in roles.values():
        for s in specs:
            for f in [io.ROOT/s['weights'],*(io.ROOT/s['design']).glob('*.py')]:plan['input_sha256'][io.relative(f)]=io.sha(f)
    plan['evaluation_roles']=roles;io.write(out/'evaluation_plan.json',plan);io.verify(plan)
    # Reuse already verified actual-damage control replays from the completed alignment experiment.
    references={}
    for f in (source/'evaluation').glob('match_*.json'):
        b=io.read(f)
        if not b['role'].startswith('tail_'):continue
        for e in b['episodes']:
            if e['seed']<plan['development_band']+2:references[(b['own'],b['foe'],e['seed'],e['seat'])]=e
    assert len(references)==384
    if io.resources()['commit_headroom_gib']<4.5:raise MemoryError('Evaluation startup headroom')
    t=time.perf_counter();records=[];cells=set()
    replay_source=io.ROOT/'runs/league_rear_alignment_20261012'
    assert io.read(replay_source/'completion.json')['status']=='rear_alignment_complete'
    assert io.read(replay_source/'review.json')['status']=='paired_alignment_review_complete'
    for file in sorted((replay_source/'evaluation').glob('match_*.json')):
        b=io.read(file)
        if not b['role'].startswith('baseline'):continue
        assert b['own'] in [s['id'] for s in roles[b['role']]]
        for e in b['episodes']:
            key=(b['role'],b['own'],b['foe'],e['seed'],e['seat']);assert key not in cells;cells.add(key)
            ref=references[(b['own'],b['foe'],e['seed'],e['seat'])]
            for k in ('result','own_health','opp_health','duration'):assert e[k]==ref[k]
            assert e['result']==classify(e['own_health'],e['opp_health'])
            assert abs(e['rear_wez_seconds']-ref['tail_attack_seconds'])<1e-12
        plan['input_sha256'][io.relative(file)]=io.sha(file)
        io.write(out/'evaluation'/f'match_{1000+len(records):04d}.json',b);records.append(b)
    assert len(cells)==384
    io.write(out/'evaluation_plan.json',plan);io.verify(plan)
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(match,s,f,plan['development_band'],4):(i,role) for i,(role,s,f) in enumerate((role,s,f) for role,ss in roles.items() if not role.startswith('baseline') for s in ss for f in plan['development_opponents'])}
        for future in as_completed(jobs):
            i,role=jobs[future];b=dict(role=role,**future.result());io.write(out/'evaluation'/f'match_{i:04d}.json',b);records.append(b)
            for e in b['episodes']:
                key=(role,b['own'],b['foe'],e['seed'],e['seat']);assert key not in cells;cells.add(key)
                assert e['result']==classify(e['own_health'],e['opp_health'])
                if role.startswith('baseline'):
                    ref=references[(b['own'],b['foe'],e['seed'],e['seat'])]
                    for k in ('result','own_health','opp_health','duration'):assert e[k]==ref[k]
                    assert abs(e['rear_wez_seconds']-ref['tail_attack_seconds'])<1e-12
    evaluation_seconds=time.perf_counter()-t
    expected={(role,s['id'],f['id'],plan['development_band']+k,seat) for role,ss in roles.items() for s in ss for f in plan['development_opponents'] for k in (0,1) for seat in ('red','blue')}
    assert cells==expected and len(cells)==768
    profiles={}
    for role in roles:
        profiles[role]={}
        for foe in ['all','noncircler']+[f['id'] for f in plan['development_opponents']]:
            es=[e for b in records if b['role']==role and (foe=='all' or (foe=='noncircler' and b['foe']!='circler') or b['foe']==foe) for e in b['episodes']]
            profiles[role][foe]=profile(es)
    io.verify(plan);done=dict(status='rear_geometry_complete',training_steps=sum(r['steps'] for r in results.values()),reused_control_training_steps=sum(io.read(source/f'tail_s{s}/result.json')['steps'] for s in plan['seeds']),training_seconds=sum(r['learning_seconds'] for r in results.values()),training_games=sum(r['training_episodes'] for r in results.values()),evaluation_games=384,evaluation_cells=768,new_candidate_evaluation_games=384,reused_control_games=384,evaluation_seconds=evaluation_seconds,total_seconds=time.perf_counter()-started,profiles=profiles,policy_promoted=False)
    io.write(out/'raw_audit.json',dict(status='768_exact_cells_geometry_schedule_and_reused_controls_verified'))
    io.write(out/'completion.json',done);print({k:v for k,v in done.items() if k!='profiles'},flush=True);print({k:v['all'] for k,v in profiles.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:run(a.out.resolve(),a.smoke)
    except BaseException:
        if a.out.exists():io.write(a.out/'failure.json',dict(error=traceback.format_exc(),resources=io.resources()))
        raise
