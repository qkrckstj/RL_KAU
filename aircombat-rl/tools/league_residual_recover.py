"""Memory-bounded independent PPO continuations from a preserved learner.

The interrupted experiment is immutable. Both new RNGs start from the same
preserved actor, critic and optimizer, using fresh episode streams. This is
independent continuation after a shared prefix, not independent from scratch.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import gc
import os
import shutil
import time
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

from tools import league_residual_train as base
from tools.league_thread_benchmark import (
    ROOT, read, write, sha, relative, verify, resources, own_memory, environment, THREAD_KEYS)


def load_continuation(path, vec, seed, n_steps):
    model=PPO.load(path,env=vec,device='cpu',force_reset=True,seed=seed,n_steps=n_steps)
    prefix_steps=model.num_timesteps
    model.num_timesteps=0
    model.set_random_seed(seed)
    if model._last_obs is not None or model.n_envs!=vec.num_envs or model.n_steps!=n_steps:
        raise AssertionError('Continuation must start fresh episodes with the declared topology')
    return model,prefix_steps


def freeze(out,smoke):
    if (out/'plan.json').exists():raise FileExistsError('Use a new continuation folder')
    if any(os.environ.get(k)!='1' for k in THREAD_KEYS):raise ValueError('Set thread limits before Python starts')
    previous=ROOT/'runs/league_residual_ppo_20261007'
    old=read(previous/'plan.json');verify(old)
    failure=read(previous/'failure.json')
    if 'MemoryError: Commit/physical headroom below safe operating floor' not in failure['error']:
        raise ValueError('This continuation expects the recorded resource guard exit')
    # Native exit was recorded before recovery; archived PID numbers are not live state.
    events=[__import__('json').loads(line) for line in (previous/'observer_events.jsonl').read_text(encoding='utf-8').splitlines()]
    if not any(e['kind']=='learner_exited' and e['exit_code']==1 for e in events):
        raise ValueError('Predecessor termination was not observed')
    prefix=previous/'s3200/interrupted_learner.zip'
    teacher=old['teacher']
    model=PPO.load(prefix,device='cpu')
    prefix_steps=model.num_timesteps
    if prefix_steps!=read(previous/'s3200/interrupted_checkpoint.json')['step']:
        raise ValueError('Emergency checkpoint step mismatch')
    initial=base.export(out/'initial_submission',teacher,model.policy,out.name+'_shared_prefix')
    initial_parameters={k:v.detach().cpu().clone() for k,v in model.policy.state_dict().items()}
    torch.save(initial_parameters,out/'initial_state_dict.pt')
    del model,initial_parameters;gc.collect()
    plan=deepcopy(old)
    plan.update(smoke=smoke,workers=4,seeds=[3399] if smoke else [3300,3301],
        training_bands=[170000000] if smoke else [240000000,241000000],
        shared_prefix=relative(prefix),shared_prefix_steps=prefix_steps,
        shared_prefix_trained_steps=prefix_steps//4096*4096,
        discarded_partial_rollout_steps=prefix_steps%4096,initial_submission=initial,
        previous=relative(previous),chunk_steps=32768 if smoke else 2097152,
        base_chunks=1 if smoke else 4,maximum_chunks=1 if smoke else 8,
        resume_scope='Both3300/3301 copy the same actor/critic/optimizer after a common1,032,200-interaction prefix. Each resets RNG and starts disjoint fresh official episodes. Count the common prefix once for total compute. No exact interrupted-trajectory or independent-from-scratch claim.',
        execution_change='4 neural worker processes;1024 steps per environment per rollout. Same4096 total samples per PPO update as prior8x512. Longer per-environment GAE segments are an explicit protocol change.',
        source_sha256=dict(old['source_sha256']),input_sha256=dict(old['input_sha256']),environment=environment())
    plan['ppo']['n_steps']=1024
    plan['budget_rule'] += ' For continuation, the baseline rank is the better of the preserved teacher and shared-prefix checkpoint on the same development panel.'
    if smoke:
        wanted={'ace','pursuit','ddqn_s0','temporal_extend_right'}
        chosen=[(s,w) for s,w in zip(plan['opponents'],plan['probabilities']) if s['id'] in wanted]
        plan['opponents']=[s for s,_ in chosen]
        total=sum(w for _,w in chosen);plan['probabilities']=[w/total for _,w in chosen]
        plan['development_opponents']=plan['opponents']
        plan['development_band']=170000010;plan['development_n']=2
    plan['groups']=base.code_groups(plan['opponents'])
    for name in ('tools/league_residual_recover.py','tests/test_league_residual_recover.py'):
        plan['source_sha256'][name]=sha(ROOT/name)
    paths=[previous/'plan.json',previous/'failure.json',previous/'observer_events.jsonl',prefix,
           previous/'s3200/interrupted_checkpoint.json',previous/'development_baseline.json',
           out/'initial_state_dict.pt']+list((out/'initial_submission').glob('*'))
    if not smoke:
        check=ROOT/'runs/league_residual_recovery_smoke_20261007'
        if read(check/'completion.json')['status']!='continuation_integration_passed':
            raise ValueError('Real continuation smoke required')
        if read(check/'plan.json')['source_sha256']['tools/league_residual_recover.py']!=sha(Path(__file__)):
            raise ValueError('Recovery driver changed after integration')
        paths += [check/'plan.json',check/'completion.json',check/'s3399/continuation_smoke.json']
    for path in paths:
        if path.is_file():plan['input_sha256'][relative(path)]=sha(path)
    verify(plan);write(out/'plan.json',plan)
    for name in plan['source_sha256']:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    return plan


def train_seed(out,plan,seed,ordinal,baseline=None,initial=None):
    folder=out/f's{seed}';folder.mkdir()
    factories=[partial(base.make_worker,plan['teacher'],plan['opponents'],plan['probabilities'],
        seed*100+i,plan['training_bands'][ordinal],i,plan['workers'],'red' if i%2==0 else 'blue',plan['ppo']['gamma'])
        for i in range(plan['workers'])]
    vec=SubprocVecEnv(factories,start_method='spawn');model=None
    try:
        initialized=dict(workers=vec.env_method('worker_status'),parent=own_memory(),resources=resources())
        write(folder/'initialized_workers.json',initialized)
        if initialized['resources']['commit_headroom_gib']<2.8:raise MemoryError('Insufficient4-worker startup headroom')
        model,prefix_steps=load_continuation(ROOT/plan['shared_prefix'],vec,seed,plan['ppo']['n_steps'])
        expected=torch.load(out/'initial_state_dict.pt',map_location='cpu',weights_only=True)
        if prefix_steps!=plan['shared_prefix_steps'] or not all(torch.equal(v,expected[k]) for k,v in model.policy.state_dict().items()):
            raise AssertionError('Actor/critic changed during continuation restore')
        del expected
        model.set_logger(configure(str(folder/'metrics'),['csv']))
        if baseline is None:
            if plan['smoke']:
                baseline=base.evaluate(vec,plan['teacher'],plan['development_opponents'],
                    plan['development_band'],plan['development_n'],out/'development_baseline.json')
            else:
                baseline=read(ROOT/plan['previous']/'development_baseline.json')
                request=dict(spec=plan['teacher'],opponents=plan['development_opponents'],
                             band=plan['development_band'],n=plan['development_n'])
                if baseline['request']!=request:raise ValueError('Cannot reuse changed baseline conditions')
                base.validate_records(baseline['results'],plan['development_opponents'],plan['development_band'],plan['development_n'])
                write(out/'development_baseline_reuse.json',dict(source=plan['previous']+'/development_baseline.json',
                    sha256=sha(ROOT/plan['previous']/'development_baseline.json'),executed_new_games=0,
                    scope='Same preserved teacher and same development conditions; reuse is not new evidence.'))
        if initial is None:
            initial=base.evaluate(vec,plan['initial_submission'],plan['development_opponents'],
                plan['development_band'],plan['development_n'],out/'shared_prefix_development.json')
        best=dict(spec=plan['teacher'],rank=baseline['rank'],step=0)
        if tuple(initial['rank'])>tuple(best['rank']):best=dict(spec=plan['initial_submission'],rank=initial['rank'],step=0)
        baseline_rank=best['rank'];history=[];limit=plan['base_chunks'];block=0;extended=False
        progress=base.Progress(folder,seed)
        while block<limit:
            started=time.perf_counter()
            model.learn(total_timesteps=plan['chunk_steps'],callback=progress,reset_num_timesteps=False)
            learning_seconds=time.perf_counter()-started
            ck=folder/f'checkpoints/step_{model.num_timesteps}'
            spec=base.save_checkpoint(model,ck,plan['teacher'],f'{out.name}_s{seed}_t{model.num_timesteps}')
            record=base.evaluate(vec,spec,plan['development_opponents'],plan['development_band'],
                plan['development_n'],ck/'development.json')
            if tuple(record['rank'])>tuple(best['rank']):best=dict(spec=spec,rank=record['rank'],step=model.num_timesteps)
            row=dict(chunk=block+1,steps=model.num_timesteps,shared_prefix_steps=prefix_steps,
                learning_seconds=learning_seconds,checkpoint=relative(ck),development=relative(ck/'development.json'),
                candidate=dict(spec=spec,rank=record['rank']),retained=best,episodes=progress.completed,
                opponent_episode_counts=progress.foes.copy(),training_outcomes=progress.counts.copy(),
                evaluation_seconds=record['wall_seconds'])
            history.append(row);write(folder/f'chunk_{block+1:02d}.json',row);block+=1
            if block==plan['base_chunks']:
                previous=history[max(0,plan['base_chunks']//2-1)]['retained']['rank']
                extended=not plan['smoke'] and base.extend_budget(previous,best['rank'],baseline_rank)
                if extended:limit=plan['maximum_chunks']
                write(folder/'budget_decision.json',dict(extend=extended,limit_chunks=limit,
                    previous_rank=previous,latest_rank=best['rank'],baseline_rank=baseline_rank,
                    rule=plan['budget_rule'],final_data_used=False))
            progress.report('checkpoint_complete')
        result=dict(status='training_complete',seed=seed,selected=best,history=history,
            environment_steps=model.num_timesteps,shared_prefix_steps=prefix_steps,
            training_episodes=progress.completed,training_outcomes=progress.counts,
            opponent_episode_counts=progress.foes,minimum_commit_headroom_gib=progress.minimum_commit,
            extended=extended,resume_scope=plan['resume_scope'])
        write(folder/'result.json',result)
        if plan['smoke']:
            restored=PPO.load(ck/'learner.zip',device='cpu')
            x=model.rollout_buffer.observations.reshape(-1,40)[:128]
            if not np.array_equal(model.predict(x,deterministic=True)[0],restored.predict(x,deterministic=True)[0]):
                raise AssertionError('Trained continuation checkpoint changed actions on restore')
            write(folder/'continuation_smoke.json',dict(restored_actions_equal=True,parameters_preserved_on_load=True,
                additional_steps=model.num_timesteps,shared_prefix_steps=prefix_steps,
                workers=plan['workers'],n_steps=model.n_steps,rollout_samples=model.n_steps*model.n_envs))
        return result,baseline,initial
    except BaseException:
        if model is not None:
            model.save(folder/'interrupted_learner.zip')
            write(folder/'interrupted_checkpoint.json',dict(additional_step=model.num_timesteps,shared_prefix_steps=plan['shared_prefix_steps'],scope=plan['resume_scope']))
        raise
    finally:vec.close()


def run(out,smoke):
    torch.set_num_threads(1)
    before=resources()
    if before['commit_headroom_gib']<4.5 or before['available_memory_gib']<4.:
        raise MemoryError('Insufficient4-worker startup headroom')
    plan=freeze(out,smoke)
    write(out/'runtime.json',dict(pid=os.getpid(),workers=4,device='cpu',
        started_at=datetime.now(timezone.utc).isoformat(),resources=before,smoke=smoke))
    started=time.perf_counter();searches=[];baseline=initial=None
    for ordinal,seed in enumerate(plan['seeds']):
        verify(plan)
        current,baseline,initial=train_seed(out,plan,seed,ordinal,baseline,initial)
        searches.append(current)
    if smoke:result=dict(status='continuation_integration_passed',searches=searches,final_opened=False,heldout_opened=False)
    else:result=base.selection(out,plan,searches)
    verify(plan)
    new_steps=sum(s['environment_steps'] for s in searches)
    result.update(wall_seconds=time.perf_counter()-started,additional_environment_steps=new_steps,
        shared_prefix_steps_counted_once=plan['shared_prefix_steps'],total_environment_steps_including_shared_prefix=plan['shared_prefix_steps']+new_steps,
        completed_at=datetime.now(timezone.utc).isoformat(),resources_after=resources())
    write(out/'completion.json',result)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    try:run(args.out.resolve(),args.smoke)
    except BaseException:
        write(args.out/'failure.json',dict(error=traceback.format_exc(),pid=os.getpid(),at=datetime.now(timezone.utc).isoformat()))
        raise
