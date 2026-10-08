"""Bounded mixed-archive PPO with a preserved public-state controller prior.

Separate training RNGs, sparse development, frozen selection, fresh final ICs.
The official physics/verdicts and all previous policy artifacts are unchanged.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.league.residual_worker import make_worker, isolated_rng
from experiments.league.residual_ppo_policy import ResidualPriorPolicy, actor_parameters
from experiments.league.residual_features import numpy_logits
from tools.league_residual_export import export
from tools.league_thread_benchmark import (
    ROOT, read, write, sha, relative, verify, resources, own_memory, environment, THREAD_KEYS)
from tools.league_matches import summary
from tools.league_train import unique_entrants
from tools.league_tournament_metrics import code_groups, group_weights, profile, ranking
from tools.league_temporal_repair_train import decision, claim


def development_panel(opponents, groups, temporal, teacher, size=24):
    ids=[s['id'] for s in opponents]
    selected=set(temporal+[teacher])
    for group in sorted(set(groups.values())):
        if not any(groups[name]==group for name in selected):
            selected.add(next(name for name in ids if groups[name]==group))
    target=max(size,len(selected))
    for name in ids:
        if len(selected)>=target: break
        selected.add(name)
    return [s for s in opponents if s['id'] in selected]


def extend_budget(previous, latest, baseline):
    return bool(latest[0]-previous[0]>=.005-1e-12 and
                latest[0]-baseline[0]>=.005-1e-12 and latest[1]>=previous[1]-1e-12)


def validate_records(results, opponents, band, n):
    if len(results)!=len(opponents) or {r['foe'] for r in results}!={s['id'] for s in opponents}:
        raise ValueError('Incomplete evaluation opponent panel')
    expected={(band+i,seat) for i in range(n//2) for seat in ('red','blue')}
    for record in results:
        rows=record['episodes']
        if len(rows)!=n or {(r['seed'],r['seat']) for r in rows}!=expected:
            raise ValueError('Incomplete shared IC/seat pairs')
        if record['summary']!=summary(rows): raise ValueError('Raw summary mismatch')


def evaluate(vec, own, opponents, band, n, path):
    if path.exists(): raise FileExistsError('Do not repeat/overwrite evaluation')
    before=vec.env_method('worker_status')
    started=time.perf_counter()
    with isolated_rng():
        partitions=vec.env_method('evaluate_partition',own,opponents,band,n)
    results=[r for part in partitions for r in part]
    by_id={r['foe']:r for r in results}
    results=[by_id[s['id']] for s in opponents]
    validate_records(results,opponents,band,n)
    after=vec.env_method('worker_status')
    for a,b in zip(before,after):
        for key in ('worker_index','episodes_started','opponent_draws','episode'):
            if a[key]!=b[key]: raise AssertionError('Evaluation changed training progression')
    p=profile(results,code_groups(opponents))
    record=dict(request=dict(spec=own,opponents=opponents,band=band,n=n),results=results,
                profile=p,rank=list(ranking(p)),wall_seconds=time.perf_counter()-started,
                games=len(opponents)*n,steps=sum(e['steps'] for r in results for e in r['episodes']),
                scope='Greedy exported NumPy policy, shared ICs and both seats, unchanged official verdicts.')
    write(path,record)
    return record


def assert_actor_agreement(model):
    x=np.asarray(model.rollout_buffer.observations).reshape(-1,40)
    x=x[np.linspace(0,len(x)-1,min(1024,len(x)),dtype=int)].copy()
    parameters=actor_parameters(model.policy)
    with torch.no_grad():
        logits=model.policy.get_distribution(torch.as_tensor(x)).distribution.logits.cpu().numpy()
    actual=numpy_logits(x,parameters)
    # Torch Categorical.logits are normalized log probabilities.
    expected=logits-logits[:,:1]; observed=actual-actual[:,:1]
    error=float(np.max(np.abs(expected-observed)))
    if error>2e-5 or not np.array_equal(np.argmax(logits,axis=1),np.argmax(actual,axis=1)):
        raise AssertionError('Learned Torch/NumPy actions disagree on collected states')
    return dict(states=len(x),greedy_actions_equal=True,max_relative_logit_error=error)


def save_checkpoint(model, folder, teacher, identity):
    if folder.exists(): raise FileExistsError('Checkpoint already exists')
    folder.mkdir(parents=True)
    model.save(folder/'learner.zip')
    torch.save(dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()),folder/'rng.pt')
    spec=export(folder/'submission',teacher,model.policy,identity)
    check=assert_actor_agreement(model)
    write(folder/'checkpoint.json',dict(step=model.num_timesteps,entrant=spec,
        agreement=check,learner_sha256=sha(folder/'learner.zip'),
        resume_scope='SB3 actor/critic/optimizer and RNG saved. JSBSim mid-episode state is not serialized; continuation requires explicitly fresh episode streams, not an exact interrupted-trajectory claim.'))
    return spec


class Progress(BaseCallback):
    def __init__(self, folder, seed):
        super().__init__();self.folder=folder;self.seed=seed
        self.started=time.perf_counter();self.last_report=0.;self.completed=0
        self.counts={};self.foes={};self.steps_in_completed_episodes=0
        self.next_memory_step=0;self.minimum_commit=float('inf')

    def _on_step(self):
        rows=[]
        for done,info in zip(self.locals['dones'],self.locals['infos']):
            if done:
                meta=info['league_episode'];outcome=info['outcome']
                row=dict(**meta,outcome=outcome,won=info['won'],own_health=info['own_health'],
                         opp_health=info['opp_health'],
                         official_episode_flags=info['official_episode_flags'],**info['episode'])
                # Monitor's t is elapsed wall time; preserve official clock separately.
                row['official_t']=info['t']
                rows.append(row);self.completed+=1
                self.counts[outcome]=self.counts.get(outcome,0)+1
                self.foes[meta['opponent_id']]=self.foes.get(meta['opponent_id'],0)+1
                self.steps_in_completed_episodes+=info['episode']['l']
        if rows:
            with (self.folder/'training_episodes.jsonl').open('a',encoding='utf-8') as stream:
                for row in rows: stream.write(json.dumps(row)+'\n')
        if self.num_timesteps>=self.next_memory_step:
            self.next_memory_step=self.num_timesteps+8192
            sample=resources();self.minimum_commit=min(self.minimum_commit,sample['commit_headroom_gib'])
            if sample['commit_headroom_gib']<1.8 or sample['available_memory_gib']<1.5:
                raise MemoryError('Commit/physical headroom below safe operating floor')
        now=time.perf_counter()
        if now-self.last_report>=30:
            self.last_report=now;self.report('training')
        return True

    def report(self,stage):
        row=dict(stage=stage,seed=self.seed,steps=self.model.num_timesteps,
            episodes=self.completed,outcomes=self.counts,opponent_episode_counts=self.foes,
            steps_in_completed_episodes=self.steps_in_completed_episodes,
            elapsed_seconds=time.perf_counter()-self.started,minimum_commit_headroom_gib=self.minimum_commit,
            resources=resources(),at=datetime.now(timezone.utc).isoformat())
        write(self.folder/'progress.json',row)
        with (self.folder/'telemetry.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        print(f"{stage}: RNG{self.seed}, {row['steps']} steps, {self.completed} episodes, {row['elapsed_seconds']:.1f}s",flush=True)


def train_seed(out,plan,seed,ordinal,baseline=None):
    folder=out/f's{seed}';folder.mkdir()
    workers=plan['workers'];band=plan['training_bands'][ordinal]
    factories=[partial(make_worker,plan['teacher'],plan['opponents'],plan['probabilities'],
        seed*100+index,band,index,workers,'red' if index%2==0 else 'blue',plan['ppo']['gamma'])
        for index in range(workers)]
    vec=SubprocVecEnv(factories,start_method='spawn')
    model=None
    try:
        initialized=dict(workers=vec.env_method('worker_status'),parent=own_memory(),resources=resources())
        write(folder/'initialized_workers.json',initialized)
        if initialized['resources']['commit_headroom_gib']<2.3:
            raise MemoryError('Insufficient headroom after worker initialization')
        model=PPO(ResidualPriorPolicy,vec,seed=seed,device='cpu',verbose=0,
                  policy_kwargs=dict(prior_bias=6.),**plan['ppo'])
        model.set_logger(configure(str(folder/'metrics'),['csv']))
        progress=Progress(folder,seed)
        if baseline is None:
            baseline=evaluate(vec,plan['teacher'],plan['development_opponents'],
                plan['development_band'],plan['development_n'],out/'development_baseline.json')
        baseline_rank=baseline['rank']
        best=dict(spec=plan['teacher'],rank=baseline_rank,step=0)
        history=[];extended=False;limit=plan['base_chunks']
        block=0
        while block<limit:
            started=time.perf_counter()
            model.learn(total_timesteps=plan['chunk_steps'],callback=progress,reset_num_timesteps=False)
            learning_seconds=time.perf_counter()-started
            ck=folder/f'checkpoints/step_{model.num_timesteps}'
            spec=save_checkpoint(model,ck,plan['teacher'],f'{out.name}_s{seed}_t{model.num_timesteps}')
            record=evaluate(vec,spec,plan['development_opponents'],plan['development_band'],
                            plan['development_n'],ck/'development.json')
            if tuple(record['rank'])>tuple(best['rank']):
                best=dict(spec=spec,rank=record['rank'],step=model.num_timesteps)
            row=dict(chunk=block+1,steps=model.num_timesteps,learning_seconds=learning_seconds,
                development=relative(ck/'development.json'),checkpoint=relative(ck),
                candidate=dict(spec=spec,rank=record['rank']),retained=best,
                episodes=progress.completed,opponent_episode_counts=progress.foes.copy(),
                training_outcomes=progress.counts.copy(),evaluation_seconds=record['wall_seconds'])
            history.append(row);write(folder/f'chunk_{block+1:02d}.json',row)
            block+=1
            if block==plan['base_chunks']:
                previous=history[max(0,plan['base_chunks']//2-1)]['retained']['rank']
                extended=(not plan['smoke'] and extend_budget(previous,best['rank'],baseline_rank))
                if extended: limit=plan['maximum_chunks']
                write(folder/'budget_decision.json',dict(extend=extended,limit_chunks=limit,
                    previous_rank=previous,latest_rank=best['rank'],baseline_rank=baseline_rank,
                    rule=plan['budget_rule'],final_data_used=False))
            progress.report('checkpoint_complete')
        result=dict(status='training_complete',seed=seed,selected=best,history=history,
            environment_steps=model.num_timesteps,training_episodes=progress.completed,
            training_outcomes=progress.counts,opponent_episode_counts=progress.foes,
            minimum_commit_headroom_gib=progress.minimum_commit,extended=extended)
        write(folder/'result.json',result)
        if plan['smoke']:
            # Meaningful learner save/load check after real PPO updates.
            loaded=PPO.load(ck/'learner.zip',device='cpu')
            x=model.rollout_buffer.observations.reshape(-1,40)[:128]
            if not np.array_equal(model.predict(x,deterministic=True)[0],loaded.predict(x,deterministic=True)[0]):
                raise AssertionError('Trained PPO restore changed actions')
            if not float(torch.abs(model.policy.action_net.weight).max())>0:
                raise AssertionError('PPO did not update the actor')
            write(folder/'learner_smoke.json',dict(actor_updated=True,restored_actions_equal=True,
                environment_steps=model.num_timesteps,actor_max_abs_weight=float(torch.abs(model.policy.action_net.weight).max().detach())))
        return result,baseline
    except BaseException:
        if model is not None:
            model.save(folder/'interrupted_learner.zip')
            write(folder/'interrupted_checkpoint.json',dict(step=model.num_timesteps,
                scope='Emergency learner state; environment trajectories are not serialized. See failure.json.'))
        raise
    finally:
        vec.close()


def freeze(out,smoke,workers):
    if (out/'plan.json').exists(): raise FileExistsError('Use a new run folder; no implicit partial resume')
    if any(os.environ.get(k)!='1' for k in THREAD_KEYS): raise ValueError('Set pre-import thread limits')
    old_root=ROOT/'runs/league_pursuit_context_train_20261007'
    old=read(old_root/'plan.json');verify(old)
    qualification=ROOT/'runs/league_residual_qualify_20261007'
    qualified=read(qualification/'plan.json');verify(qualified)
    if read(qualification/'completion.json')['status']!='actual_flight_qualification_passed':
        raise ValueError('Full-flight qualification required')
    extra=[read(old_root/f's{s}/result.json')['selected']['spec'] for s in (3100,3101)]
    extra.append(read(old_root/'s3100/g3/development/candidate_000.json')['request']['spec'])
    opponents=unique_entrants(old['opponents']+extra)
    if smoke:
        wanted={'ace','pursuit','ddqn_s0','temporal_extend_right'}
        opponents=[s for s in opponents if s['id'] in wanted]
    groups=code_groups(opponents);balanced=group_weights(groups)
    temporal=[s['id'] for s in opponents if s['id'] in old['temporal_ids']]
    weak_total=sum(old['weakness'].get(s['id'],0.) for s in opponents)
    probabilities=[.4/len(opponents)+.3*balanced[s['id']]+.3*old['weakness'].get(s['id'],0.)/weak_total for s in opponents]
    dev=opponents if smoke else development_panel(opponents,groups,temporal,old['warm_start']['id'])
    sources=dict(qualified['source_sha256']);inputs=dict(qualified['input_sha256'])
    for name in ('tools/league_residual_train.py','tests/test_league_residual_train.py'):
        sources[name]=sha(ROOT/name)
    paths=[old_root/'plan.json',qualification/'plan.json',qualification/'completion.json']
    for spec in extra:
        if spec['kind']=='submission': paths+=list((ROOT/spec['design']).glob('*.py'))+[ROOT/spec['weights']]
    if not smoke:
        smoke_root=ROOT/'runs/league_residual_smoke_20261007'
        if read(smoke_root/'completion.json')['status']!='learner_integration_passed':
            raise ValueError('Real PPO update/checkpoint integration smoke required')
        paths += [smoke_root/'plan.json',smoke_root/'completion.json',smoke_root/'s3299/learner_smoke.json']
        if read(smoke_root/'plan.json')['source_sha256']['tools/league_residual_train.py']!=sha(Path(__file__)):
            raise ValueError('Training driver changed after integration smoke')
    for path in paths:inputs[relative(path)]=sha(path)
    plan=dict(teacher=old['warm_start'],opponents=opponents,groups=groups,temporal_ids=temporal,
        probabilities=probabilities,opponent_mixture='Alternate full shuffled coverage with frozen weighted draws; effective mixture is50%uniform+.5*(40%uniform+30%codegroup+30%known_temporal_weakness).',
        development_opponents=dev,workers=workers,seeds=[3299] if smoke else [3200,3201],
        training_bands=[170000000] if smoke else [230000000,231000000],
        development_band=170000010 if smoke else 66000000,development_n=2 if smoke else 4,
        selection_band=66010000,selection_n=40,final_band=67000000,final_n=80,
        heldout_band=68000000,heldout_n=80,heldout_opponents=old['heldout']['opponents'],
        chunk_steps=32768 if smoke else 2097152,base_chunks=1 if smoke else 4,maximum_chunks=1 if smoke else 8,
        ppo=dict(learning_rate=1e-4,n_steps=512,batch_size=512,n_epochs=4,gamma=.999,
            gae_lambda=.99,clip_range=.1,target_kl=.015,ent_coef=.005,max_grad_norm=.5),
        budget_rule='At4chunks(8,388,608steps), extend to8chunks(16,777,216) only if retained development mean(uniform,codegroup) win rate improved >=.005 since2chunks and exceeds baseline by>=.005, without a lower retained lower-quarter score. No selection/final results used.',
        selection_rule='Retain the best sparse-development checkpoint per independent RNG, including unchanged teacher. Evaluate unique challengers and teacher on all archived foes; require existing aggregate profile plus temporal mean score gain>=.025 and extend-right gain>=.10. Freeze one choice before final.',
        final_rule='Same existing repair profile plus positive lower95% IC-cluster win-gain bounds for uniform and codegroup weights. Holdout only after final passes; parameter variations of known temporal families, not unseen architecture.',
        reward='Learning-only terminal utility kill1+.05*survivingHP,loss-1,draw0 plus gamma*Phi(next)-Phi(current), terminalPhi0. Official timeouts are finite-horizon terminals. Official verdicts remain unchanged.',
        smoke=smoke,environment=environment(),source_sha256=sources,input_sha256=inputs,
        scope='CPU PPO actor/critic updates against frozen diverse archive. Existing teacher and official rules preserved. No GitHub publication. Neural rollout workers replace the completed CEM pool; no overlapping simulation pools.')
    if plan['chunk_steps']%(workers*plan['ppo']['n_steps']): raise ValueError('Budget must be full rollouts')
    verify(plan);write(out/'plan.json',plan)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    return plan


def selection(out,plan,searches):
    challengers=unique_entrants([s['selected']['spec'] for s in searches])
    challengers=[s for s in challengers if s!=plan['teacher']]
    result=dict(status='no_development_improvement',searches=searches,
                final_opened=False,heldout_opened=False,retained=plan['teacher'])
    if not challengers:return result
    # The training workers were closed. Reuse the same qualified ordinary-worker
    # RPC path, allocating only one pool at a time.
    factories=[partial(make_worker,plan['teacher'],plan['opponents'],plan['probabilities'],
        320099+i,232000000,i,plan['workers'],'red' if i%2==0 else 'blue') for i in range(plan['workers'])]
    vec=SubprocVecEnv(factories,start_method='spawn')
    try:
        candidates=[plan['teacher']]+challengers
        rows=[evaluate(vec,spec,plan['opponents'],plan['selection_band'],plan['selection_n'],
                       out/f'selection/candidate_{i:03d}.json') for i,spec in enumerate(candidates)]
        comparisons=[decision(r,rows[0],plan) for r in rows[1:]]
        eligible=[i+1 for i,c in enumerate(comparisons) if c['repair_passed']]
        chosen_index=max(eligible,key=lambda i:tuple(rows[i]['rank'])) if eligible else 0
        frozen=dict(chosen=candidates[chosen_index],chosen_index=chosen_index,eligible=eligible,
                    comparisons=comparisons,selection_band=plan['selection_band'])
        write(out/'frozen_selection.json',frozen)
        result.update(status='selection_failed',selection=frozen)
        if not eligible:return result
        claim(out,plan,plan['final_band']);result['final_opened']=True
        final=[evaluate(vec,spec,plan['opponents'],plan['final_band'],plan['final_n'],
                        out/f'final/candidate_{i:03d}.json') for i,spec in enumerate([plan['teacher'],frozen['chosen']])]
        comparison=decision(final[1],final[0],plan,final=True)
        write(out/'final/comparison.json',comparison);result['final_comparison']=comparison
        if not comparison['repair_passed']:
            result['status']='final_profile_failed';return result
        result.update(status='final_profile_passed',retained=frozen['chosen'])
        claim(out,plan,plan['heldout_band']);result['heldout_opened']=True
        held=[evaluate(vec,spec,plan['heldout_opponents'],plan['heldout_band'],plan['heldout_n'],
                       out/f'heldout/candidate_{i:03d}.json') for i,spec in enumerate([plan['teacher'],frozen['chosen']])]
        from tools.league_tournament_metrics import paired_gain
        write(out/'heldout/comparison.json',dict(candidate=held[1]['profile'],reference=held[0]['profile'],
            paired_gain=paired_gain(held[1]['results'],held[0]['results'],{s['id']:1/len(plan['heldout_opponents']) for s in plan['heldout_opponents']}),
            scope='Fresh initial conditions and unused parameter combinations of known temporal families; no selection on these results.'))
        return result
    finally: vec.close()


def run(out,smoke,workers):
    torch.set_num_threads(1)
    plan=freeze(out,smoke,workers)
    before=resources()
    if before['commit_headroom_gib']<5.8 or before['available_memory_gib']<5.:
        raise MemoryError('Insufficient neural-worker startup memory headroom')
    write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
        workers=workers,device='cpu',resources=before,mode='smoke' if smoke else 'training'))
    started=time.perf_counter();searches=[];baseline=None
    for ordinal,seed in enumerate(plan['seeds']):
        verify(plan)
        current,baseline=train_seed(out,plan,seed,ordinal,baseline)
        searches.append(current)
    if smoke:result=dict(status='learner_integration_passed',searches=searches,final_opened=False,heldout_opened=False)
    else:result=selection(out,plan,searches)
    verify(plan)
    result.update(wall_seconds=time.perf_counter()-started,environment_steps=sum(s['environment_steps'] for s in searches),
                  completed_at=datetime.now(timezone.utc).isoformat(),resources_after=resources())
    write(out/'completion.json',result)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--workers',type=int,default=8,choices=[4,8])
    args=parser.parse_args()
    try:run(args.out.resolve(),args.smoke,args.workers)
    except BaseException:
        write(args.out/'failure.json',dict(error=traceback.format_exc(),pid=os.getpid(),at=datetime.now(timezone.utc).isoformat()))
        raise
