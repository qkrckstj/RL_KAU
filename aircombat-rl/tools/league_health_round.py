"""One versioned health-rule round: common starting screen, two PPO continuations.

No legacy physics/source changes, automatic publication, or fabricated promotion.
Top-level imports stay Torch-free so qualified NumPy evaluation workers stay pure.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from argparse import ArgumentParser
from collections import Counter,defaultdict
from concurrent.futures import as_completed
from datetime import datetime,timezone
from functools import partial
from pathlib import Path
import gc,json,shutil,time,traceback
from tools import league_thread_benchmark as io

RULE='user_terminal_health_round4_v2'


def freeze(out,smoke):
    if out.exists():raise FileExistsError(out)
    source=io.ROOT/'runs/league_compact_rotated_train_20261008'
    old=io.read(source/'plan.json');io.verify(old)
    compact=io.read(io.ROOT/'runs/league_compact_ppo_train_20261008/completion.json')
    rotated=io.read(source/'completion.json')
    candidates={'baseline':old['warm_start'],
                'compact':next(s for s in compact['searches'] if s['seed']==7800)['history'][0]['candidate'],
                'rotated':next(s for s in rotated['searches'] if s['seed']==7801)['history'][0]['candidate']}
    roles={k:[a['spec'] for a in v['replicas']] for k,v in candidates.items()}
    roles.update(cem=[next(s for s in old['opponents'] if s['id']=='extend_cem_s6800_g3_c9')],
                 protected_cem=[next(s for s in old['opponents'] if s['id']=='intercept_s2500_g3_c6')],teacher=[old['teacher']])
    from tools.league_tournament_metrics import code_groups
    panel=old['development_opponents'];groups=code_groups(panel)
    mandatory=['ace','evader','temporal_extend_left','untrained_parameter_01']
    selected=[s for s in panel if s['id'] in mandatory]
    if smoke:selected=selected[:2]
    else:
        while len(selected)<24:
            used=Counter(groups[s['id']] for s in selected)
            selected.append(min((s for s in panel if s not in selected),key=lambda s:(used[groups[s['id']]],s['id'])))
    profile=io.read(io.ROOT/'experiments/league/execution_profile_budget_20261008.json')
    plan=dict(rule=RULE,smoke=smoke,teacher=old['teacher'],opponents=old['opponents'],probabilities=old['probabilities'],
        coverage_period=old['coverage_period'],ppo=old['ppo'],source_ppo_configuration=old['source_ppo_configuration'],
        candidates=candidates,roles=roles,development_opponents=selected,groups=code_groups(selected),
        processes=2 if smoke else 8,workers=8 if smoke else 32,
        seeds=[8899] if smoke else [8900,8901],training_bands=[170630000] if smoke else [307000000,308000000],
        development_band=170640000 if smoke else 114000000,development_n=2 if smoke else 16,
        chunk_steps=20480 if smoke else 1310720,execution_profile=profile,
        source_sha256=dict(old['source_sha256']),input_sha256=dict(old['input_sha256']),
        scope='User selected8physical32virtual workers, cancelling worker benchmark. Health adjudication/reward changed;48foe roster and probabilities frozen from prior run to isolate reward migration. Starting PPO selected on common new-rule development; CEM preserved as operational comparator. Two continuation RNGs share learned ancestor. No global promotion or GitHub upload.',
        decision_rule='After both pilots: consider one extra chunk if mean win improves>=.02,group-balanced win improves and lower-quarter win falls<=.03. Otherwise assess uncertainty/regression then change one factor in a new frozen run. Selection consumes this development panel, not heldout evidence.')
    names=['experiments/league/health_rule_v2.py','tools/league_health_round.py','tests/test_health_training_v2.py']
    for name in names:plan['source_sha256'][name]=io.sha(io.ROOT/name)
    for p in [source/'plan.json',source/'completion.json',io.ROOT/'runs/league_compact_ppo_train_20261008/completion.json',io.ROOT/'runs/health_v2_checks_20261008.xml']:
        plan['input_sha256'][io.relative(p)]=io.sha(p)
    for c in candidates.values():plan['input_sha256'][c['learner']]=io.sha(io.ROOT/c['learner'])
    for specs in list(roles.values())+[selected]:
        for spec in specs:
            if spec['kind']=='submission':
                for p in list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]:plan['input_sha256'][io.relative(p)]=io.sha(p)
    claims=[(f'training_reservation_{b}.json',b,1000000) for b in plan['training_bands']]
    claims.append((f'sampled_assessment_reservation_{plan["development_band"]}.json',plan['development_band'],plan['development_n']//2))
    if any((io.ROOT/'runs'/n).exists() for n,_,_ in claims):raise FileExistsError('Condition reservation already exists')
    for name,start,count in claims:io.write(io.ROOT/'runs'/name,dict(run=io.relative(out),start=start,stop_exclusive=start+count,rule=RULE))
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in plan['source_sha256']:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,dest)
    return plan


def summarize(records,groups):
    from experiments.league.health_rule_v2 import classify
    foes=defaultdict(list)
    for record in records:
        for row in record['episodes']:foes[record['foe']].append(classify(row['own_health'],row['opp_health']))
    assert set(foes)==set(groups)
    counts={k:dict(Counter(v),games=len(v)) for k,v in foes.items()}
    rates={k:counts[k].get('win',0)/counts[k]['games'] for k in foes}
    grouped=defaultdict(list)
    for k,v in rates.items():grouped[groups[k]].append(v)
    rates_sorted=sorted(rates.values());tail=max(1,len(rates_sorted)//4)
    return dict(mean_win_rate=sum(rates.values())/len(rates),group_balanced_win_rate=sum(sum(v)/len(v) for v in grouped.values())/len(grouped),
                lower_quarter_win_rate=sum(rates_sorted[:tail])/tail,worst_win_rate=min(rates_sorted),opponents=counts,
                games=sum(len(v) for v in foes.values()),wins=sum(v.count('win') for v in foes.values()),
                draws=sum(v.count('draw') for v in foes.values()),losses=sum(v.count('loss') for v in foes.values()))


def rank(profile):
    return (.5*(profile['mean_win_rate']+profile['group_balanced_win_rate']),profile['lower_quarter_win_rate'],profile['worst_win_rate'])


def evaluate(plan,roles,folder):
    from tools.league_split_pool import SplitPool
    from tools.league_matches import duel
    from experiments.league.health_rule_v2 import classify
    folder.mkdir();profile=plan['execution_profile'];specs={s['id']:s for values in roles.values() for s in values}
    jobs=[dict(own=s,foe=f,band=plan['development_band'],n=plan['development_n']) for s in specs.values() for f in plan['development_opponents']]
    io.write(folder/'request.json',dict(rule=RULE,roles=roles,jobs=jobs))
    results=[];started=time.perf_counter()
    if io.resources()['commit_headroom_gib']<4.5:raise MemoryError('Evaluation startup headroom')
    with io.Sampler() as sampler:
        with SplitPool(4 if plan['smoke'] else profile['workers'],1 if plan['smoke'] else profile['neural_workers'],
                       profile['approved_signatures'],profile['pure_warmup'],profile['neural_warmup']) as pool:
            io.write(folder/'initialized_workers.json',pool.initialized_workers)
            pending={pool.submit(duel,j['own'],j['foe'],j['band'],j['n']):(i,j) for i,j in enumerate(jobs)}
            for future in as_completed(pending):
                i,job=pending[future];result=future.result()
                assert len(result['episodes'])==job['n']
                assert {(r['seed'],r['seat']) for r in result['episodes']}=={(job['band']+k,s) for k in range(job['n']//2) for s in ('red','blue')}
                for row in result['episodes']:
                    row['result']=classify(row['own_health'],row['opp_health']);row['verdict_rule']=RULE
                io.write(folder/'matches'/f'match_{i:04d}.json',dict(job=job,result=result))
                results.append(result)
                if len(results)%24==0:print('health evaluation',folder.name,len(results),'/',len(jobs),flush=True)
    summaries={role:summarize([r for r in results if r['own'] in {s['id'] for s in values}],plan['groups']) for role,values in roles.items()}
    report=dict(rule=RULE,roles=summaries,games=len(jobs)*plan['development_n'],wall_seconds=time.perf_counter()-started,resources=sampler.result)
    io.write(folder/'summary.json',report);return report


def train(plan,warm,seed,band,folder):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from experiments.league.grouped_vec_env import GroupedSubprocVecEnv
    from experiments.league.health_rule_v2 import make_health_worker
    from tools.league_residual_recover import load_continuation
    from tools.league_residual_train import save_checkpoint
    from tools.league_sampled_strategy import make_progress
    from tools.league_residual_action_mode import build_variant
    from tools.league_history_clone_matched_check import ppo_configuration
    from tools.league_sampled_ppo_continue import same_tree
    from tools.league_residual_profile import tensor_digest
    folder.mkdir();torch.set_num_threads(1);local=dict(plan,seed=seed,ic_band=band);vec=None;model=None
    if io.resources()['commit_headroom_gib']<4.5:raise MemoryError('Learning startup headroom')
    try:
        vec=GroupedSubprocVecEnv([partial(make_health_worker,local,i) for i in range(plan['workers'])],plan['processes'])
        io.write(folder/'initialized_workers.json',dict(workers=vec.env_method('profile_status'),resources=io.resources()))
        if io.resources()['commit_headroom_gib']<2.8:raise MemoryError('Learning initialized headroom')
        model,prefix=load_continuation(io.ROOT/warm['learner'],vec,seed,plan['ppo']['n_steps'])
        assert ppo_configuration(model)==plan['source_ppo_configuration']
        before=PPO.load(io.ROOT/warm['learner'],device='cpu')
        assert same_tree(before.policy.state_dict(),model.policy.state_dict()) and same_tree(before.policy.optimizer.state_dict(),model.policy.optimizer.state_dict())
        del before;gc.collect();initial=tensor_digest(model.policy.state_dict())
        model.set_logger(configure(str(folder/'metrics'),['csv']));progress=make_progress(folder,seed)
        started=time.perf_counter();model.learn(plan['chunk_steps'],callback=progress,reset_num_timesteps=False);elapsed=time.perf_counter()-started
        ck=folder/'checkpoint';identity=folder.parent.name+'_'+folder.name
        greedy=save_checkpoint(model,ck,plan['teacher'],identity)
        replicas=[dict(action_seed=a,spec=build_variant(greedy,ck/f'sampled_{a}',a,identity+f'_a{a}')) for a in (4900,4901)]
        restored=PPO.load(ck/'learner.zip',device='cpu');assert same_tree(restored.policy.optimizer.state_dict(),model.policy.optimizer.state_dict())
        result=dict(rule=RULE,seed=seed,learner=io.relative(ck/'learner.zip'),greedy=greedy,replicas=replicas,steps=model.num_timesteps,
                    learning_seconds=elapsed,training_episodes=progress.completed,training_outcomes=progress.counts,
                    initial_parameters=initial,final_parameters=tensor_digest(model.policy.state_dict()),minimum_commit_headroom_gib=progress.minimum_commit,
                    optimizer_restore_exact=True)
        assert result['initial_parameters']!=result['final_parameters']
        io.write(folder/'result.json',result);return result
    except BaseException:
        if model is not None:model.save(folder/'interrupted_learner.zip')
        raise
    finally:
        if vec is not None:vec.close()


def run(out,smoke):
    plan=freeze(out,smoke);io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    initial=evaluate(plan,plan['roles'],out/'starting_screen')
    choice=max(plan['candidates'],key=lambda k:rank(initial['roles'][k]))
    warm=plan['candidates'][choice]
    io.write(out/'starting_choice.json',dict(role=choice,checkpoint=warm,profile=initial['roles'][choice],rule='highest combined mean/group win then lower-quarter/worst on common consumed development; no global promotion'))
    results=[]
    for seed,band in zip(plan['seeds'],plan['training_bands']):
        io.verify(plan);print('health main learning start',seed,choice,plan['chunk_steps'],flush=True)
        result=train(plan,warm,seed,band,out/f's{seed}');results.append(result);io.write(out/'completed_training.json',results);gc.collect()
    roles={f's{r["seed"]}':[a['spec'] for a in r['replicas']] for r in results}
    assessed=evaluate(plan,roles,out/'trained_screen');start=initial['roles'][choice]
    profiles=list(assessed['roles'].values())
    mean=lambda key:sum(p[key] for p in profiles)/len(profiles)
    extend=(mean('mean_win_rate')>=start['mean_win_rate']+.02 and mean('group_balanced_win_rate')>start['group_balanced_win_rate']
            and all(p['lower_quarter_win_rate']>=start['lower_quarter_win_rate']-.03 for p in profiles)
            and all(p['mean_win_rate']>=start['mean_win_rate']-.02 for p in profiles))
    decision=dict(extend_same_method=extend,next_action='matched_additional_chunk_then_fresh_confirmation' if extend else 'analyze_per_opponent_and_reward_alignment_before_next_frozen_experiment',
                  baseline=start,candidate_profiles=assessed['roles'],policy_promoted=False,rule=plan['decision_rule'])
    io.write(out/'decision.json',decision);io.verify(plan)
    io.write(out/'completion.json',dict(status='health_round_complete',rule=RULE,starting_role=choice,additional_steps=sum(r['steps'] for r in results),
        training_seconds=sum(r['learning_seconds'] for r in results),evaluation_seconds=initial['wall_seconds']+assessed['wall_seconds'],
        decision=decision,smoke=smoke,github_uploaded=False))
    print('health round complete',decision['next_action'],flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:run(a.out.resolve(),a.smoke)
    except BaseException:
        io.write(a.out/'failure.json',dict(error=traceback.format_exc(),resources=io.resources()));raise
