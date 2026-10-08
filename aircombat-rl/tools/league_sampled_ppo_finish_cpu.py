"""CPU-wheel remainder continuation with measured lower startup overhead.

The completed first branch and all failed runs remain immutable. New episode
and learner RNG streams are explicit; mid-episode simulation/RNG is not restored.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime,timezone
from functools import partial
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import zipfile
from tools import league_thread_benchmark as io
from tools.league_sampled_ppo_continue import pair_rank,should_extend,evaluate,same_tree
from experiments.league.memory_pause import guarded_step,wait_for_headroom


def freeze(source,out,smoke,smoke_run):
    if out.exists():raise FileExistsError('Use a new remainder directory')
    old=io.read(source/'plan.json');io.verify(old)
    stopped=io.read(source/'termination_analysis.json')
    if stopped['exit_code']!=1 or any(io.process_live(pid) for pid in [stopped['controller_pid']]+stopped['worker_pids']):
        raise RuntimeError('Expected a verified stopped predecessor and children')
    first=io.read(source/'s4600/result.json')
    prior=io.read(source/'s4601/chunk_01.json')
    if first['additional_steps']!=4194304 or prior['steps']!=1048576:
        raise ValueError('Unexpected completed predecessor budgets')
    saved=source/'s4601/interrupted_learner.zip'
    if io.sha(saved)!=stopped['checkpoint_sha256'] or stopped['completed_rollout_interactions']!=1572864:
        raise ValueError('Unexpected interrupted second-branch learner')
    sources,inputs=dict(old['source_sha256']),dict(old['input_sha256'])
    for name in ('tools/league_sampled_ppo_finish_cpu.py','experiments/league/memory_pause.py','tests/test_league_memory_pause.py'):
        sources[name]=io.sha(io.ROOT/name)
    paths=[source/name for name in ('plan.json','failure.json','termination_analysis.json','baseline.json',
        's4600/result.json','s4601/chunk_01.json','s4601/retained_starting_baseline.json',
        's4601/interrupted_learner.zip','s4601/interrupted_checkpoint.json','s4601/guard_failure_context.json',
        's4601/training_episodes.jsonl')]
    paths += [io.ROOT/'runs/memory_pause_checks_20261007.xml']
    for selected in (first['selected'],prior['retained']):
        for spec in [selected['greedy']]+[r['spec'] for r in selected['replicas']]:
            paths+=list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]
    if not smoke:
        checked=io.read(smoke_run/'plan.json')
        if io.read(smoke_run/'completion.json')['status']!='sampled_remainder_integration_passed' or any(
                checked['source_sha256'].get(name)!=digest for name,digest in sources.items()):
            raise ValueError('Matching completed remainder integration run required')
        paths += [smoke_run/'plan.json',smoke_run/'completion.json']
    for path in paths:inputs[io.relative(path)]=io.sha(path)
    plan=dict(old)
    plan.update(previous=io.relative(source),source_sha256=sources,input_sha256=inputs,
        smoke=smoke,processes=1,workers=8,ppo=dict(old['ppo'],n_steps=512),
        resume_checkpoint=io.relative(saved),resume_rng=4799 if smoke else 4701,
        original_branch_rng=4601,training_bands=[170100000] if smoke else [266000000],
        prefix_branch_interactions=stopped['saved_interactions'],
        prefix_branch_completed_rollout_steps=stopped['completed_rollout_interactions'],
        discarded_interactions=stopped['pending_rollout_interactions_discarded_on_fresh_resume'],
        remaining_base_steps=524288,additional_budgets=[32768] if smoke else [524288,1048576,1048576],
        original_first_branch=first,original_second_first_chunk=prior,
        initial_comparator=io.read(source/'s4601/retained_starting_baseline.json'),
        development_band=170080000 if smoke else old['development_band'],
        development_n=2 if smoke else old['development_n'],environment=io.environment(),
        memory_rule='CPU2.14.1 wheel required. Pre-pool headroom2.4GiB and initialized2.0GiB; runtime1.8GiB commit/1.5GiB physical floors unchanged. One physical worker/eight virtual envs. Existing1.8GiB commit/1.5GiB physical guard PAUSES simulation and optimization. Resume after two5-second samples >=2.8GiB commit/2GiB physical. Critical<.5GiB commit/.75GiB physical or300-second timeout saves and aborts. No OS settings changed.',
        quality_scope='Composite of completed4600 and interrupted4601 continued with RNG4701/fresh ICs. Same original shared starting prefix, not from-scratch replication or exact interrupted trajectory. Prior8 pending transitions discarded; training rollout budget preserved.',
        resume_scope='Saved actor/critic/optimizer loaded exactly; emergency RNG/mid-episode JSBSim state not restored. First completed branch never retrained. Normal new checkpoints include RNG; emergency saves retain learner/optimizer only.')
    if smoke:
        wanted={'ace','ddqn_s0','temporal_extend_right','sampled_s3300_t8388608_a4900'}
        plan['development_opponents']=[s for s in plan['opponents'] if s['id'] in wanted]
        if len(plan['development_opponents'])!=4:raise ValueError('Missing integration foes')
    else:
        reservation=io.ROOT/'runs/training_reservation_266000000.json'
        if reservation.exists():raise FileExistsError('Training band already reserved')
        io.write(reservation,dict(run=io.relative(out),start=266000000,stop_exclusive=267000000))
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(io.ROOT/name,target)
    return plan


def train(out,plan):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from experiments.league.grouped_vec_env import GroupedSubprocVecEnv
    from tools import league_residual_train as base
    from tools.league_residual_profile import make_profile_worker,tensor_digest
    from tools.league_residual_recover import load_continuation
    from tools.league_residual_action_mode import build_variant
    if torch.__version__ != "2.14.1+cpu":raise ValueError("Qualified CPU wheel required")
    torch.set_num_threads(1)
    folder=out/f"s{plan['resume_rng']}";folder.mkdir()
    local=dict(plan,seed=plan['resume_rng'],ic_band=plan['training_bands'][0])
    vec,model=None,None
    try:
        before=io.resources()
        if min(before[k] for k in ('commit_headroom_gib','available_memory_gib'))<2.4:
            raise MemoryError('Insufficient one-worker startup headroom')
        vec=GroupedSubprocVecEnv([partial(make_profile_worker,local,i,False) for i in range(8)],processes=1)
        statuses=vec.env_method('profile_status')
        if any(s['cached_opponents']!=len(plan['opponents']) for s in statuses):
            raise AssertionError('Training opponent cache incomplete')
        io.write(folder/'initialized_workers.json',dict(virtual_environments=statuses,resources=io.resources()))
        if io.resources()['commit_headroom_gib']<2.0:
            raise MemoryError('Insufficient initialized learner headroom')
        model,prefix=load_continuation(io.ROOT/plan['resume_checkpoint'],vec,plan['resume_rng'],512)
        vec.seed(plan['resume_rng']*1000)
        with zipfile.ZipFile(io.ROOT/plan['resume_checkpoint']) as archive:
            expected=torch.load(BytesIO(archive.read('policy.pth')),map_location='cpu',weights_only=True)
            optimizer=torch.load(BytesIO(archive.read('policy.optimizer.pth')),map_location='cpu',weights_only=True)
        if prefix!=plan['prefix_branch_interactions'] or not same_tree(expected,model.policy.state_dict()) or not same_tree(optimizer,model.policy.optimizer.state_dict()):
            raise AssertionError('Emergency learner/optimizer changed on load')
        del expected,optimizer
        initial_digest=tensor_digest(model.policy.state_dict())
        model.set_logger(configure(str(folder/'metrics'),['csv']))
        class PausingProgress(base.Progress):
            def __init__(self):
                super().__init__(folder,plan['resume_rng'])
                self.pause_count=0;self.pause_seconds=0.
            def emit_pause(self,row):
                row=dict(row,at=datetime.now(timezone.utc).isoformat(),steps=self.model.num_timesteps)
                with (folder/'resource_pauses.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(row)+'\n')
                print(f"{row['stage']}: {row['elapsed_seconds']:.1f}s, headroom {row['resources']['commit_headroom_gib']:.3f}GiB",flush=True)
            def pause(self):
                self.pause_count+=1
                row=wait_for_headroom(io.resources,self.emit_pause)
                self.pause_seconds+=row['elapsed_seconds']
                self.report('memory_recovered')
            def _on_step(self):
                return guarded_step(super()._on_step,self.pause)
            def report(self,stage):
                super().report(stage)
                rows=vec.env_method('profile_status')
                memory={r['memory']['pid']:r['memory'] for r in rows}
                row=dict(at=datetime.now(timezone.utc).isoformat(),steps=self.model.num_timesteps,
                    parent=io.own_memory(),workers=list(memory.values()),resources=io.resources(),
                    pause_count=self.pause_count,pause_seconds=self.pause_seconds)
                with (folder/'process_memory.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(row)+'\n')
        progress=PausingProgress()
        prefix_rows=[json.loads(line) for line in (io.ROOT/plan['previous']/'s4601/training_episodes.jsonl').read_text('utf-8').splitlines()]
        old_counts=Counter(r['outcome'] for r in prefix_rows)
        old_foes=Counter(r['opponent_id'] for r in prefix_rows)
        best=plan['original_second_first_chunk']['retained']
        history=[] if plan['smoke'] else [plan['original_second_first_chunk']]
        extended=False
        for index,budget in enumerate(plan['additional_budgets']):
            if index>0 and not extended:break
            started=time.perf_counter()
            model.learn(budget,callback=progress,reset_num_timesteps=False)
            seconds=time.perf_counter()-started
            cumulative=plan['prefix_branch_completed_rollout_steps']+model.num_timesteps
            checkpoint=folder/f'checkpoints/step_{model.num_timesteps}'
            identity=f"{out.name}_s{plan['resume_rng']}_t{model.num_timesteps}"
            greedy=base.save_checkpoint(model,checkpoint,plan['teacher'],identity)
            replicas,records=[],[]
            for action_seed in plan['action_replicas']:
                spec=build_variant(greedy,checkpoint/f'sampled_{action_seed}',action_seed,identity+f'_a{action_seed}')
                replicas.append(dict(action_seed=action_seed,spec=spec))
                records.append(evaluate(vec,spec,plan['development_opponents'],plan['development_band'],plan['development_n'],checkpoint/f'development_{action_seed}.json'))
            candidate=dict(learner=io.relative(checkpoint/'learner.zip'),greedy=greedy,replicas=replicas,
                rank=pair_rank([r['profile'] for r in records]),step=cumulative,local_step=model.num_timesteps)
            if plan['smoke'] or tuple(candidate['rank'])>tuple(best['rank']):best=candidate
            row=dict(chunk=index+2,steps=cumulative,local_steps=model.num_timesteps,segment_learning_seconds=seconds,
                evaluation_seconds=sum(r['wall_seconds'] for r in records),evaluation_games=sum(r['executed_new_games'] for r in records),
                candidate=candidate,retained=best,episodes=len(prefix_rows)+progress.completed,
                training_outcomes=dict(old_counts+Counter(progress.counts)),opponent_episode_counts=dict(old_foes+Counter(progress.foes)),
                timing_scope='Learning time covers this resumed segment only; interrupted partial segment timing is excluded.')
            history.append(row);io.write(folder/f'chunk_{index+2:02d}.json',row)
            if index==0:
                extended=not plan['smoke'] and should_extend(plan['original_second_first_chunk']['retained']['rank'],best['rank'],plan['initial_comparator']['rank'])
                io.write(folder/'budget_decision.json',dict(extend=extended,base_completed_steps=cumulative,
                    maximum_completed_steps=4194304,first_chunk_rank=plan['original_second_first_chunk']['retained']['rank'],
                    latest_rank=best['rank'],initial_rank=plan['initial_comparator']['rank'],rule=plan['budget_rule']))
            progress.report('resumed_checkpoint_complete')
        result=dict(seed=4601,resume_rng=plan['resume_rng'],selected=best,history=history,
            additional_steps=plan['prefix_branch_interactions']+model.num_timesteps,
            completed_rollout_steps=plan['prefix_branch_completed_rollout_steps']+model.num_timesteps,
            local_resume_steps=model.num_timesteps,discarded_interactions=plan['discarded_interactions'],
            training_episodes=len(prefix_rows)+progress.completed,outcomes=dict(old_counts+Counter(progress.counts)),
            opponent_episode_counts=dict(old_foes+Counter(progress.foes)),extended=extended,
            initial_parameters_sha256=initial_digest,final_parameters_sha256=tensor_digest(model.policy.state_dict()),
            minimum_commit_headroom_gib=progress.minimum_commit,resource_pause_count=progress.pause_count,
            resource_pause_seconds=progress.pause_seconds,scope=plan['quality_scope'])
        if plan['smoke']:
            restored=PPO.load(checkpoint/'learner.zip',device='cpu')
            if not same_tree(restored.policy.state_dict(),model.policy.state_dict()) or not same_tree(restored.policy.optimizer.state_dict(),model.policy.optimizer.state_dict()):
                raise AssertionError('Saved learner/optimizer restore differs')
            if initial_digest==result['final_parameters_sha256']:raise AssertionError('No actual parameter update')
            result['learner_optimizer_restore_exact']=True
        io.write(folder/'result.json',result)
        return result
    except BaseException:
        if model is not None:
            model.save(folder/'interrupted_learner.zip')
            io.write(folder/'interrupted_checkpoint.json',dict(local_steps=model.num_timesteps,
                cumulative_branch_interactions=plan['prefix_branch_interactions']+model.num_timesteps,scope=plan['resume_scope']))
        raise
    finally:
        if vec is not None:vec.close()


def run(source,out,smoke,smoke_run):
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Set thread limits before import')
    plan=freeze(source,out,smoke,smoke_run)
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
        processes=1,virtual_environments=8,device='cpu',smoke=smoke))
    started=time.perf_counter();second=train(out,plan);io.verify(plan)
    searches=[second] if smoke else [plan['original_first_branch'],second]
    io.write(out/'completion.json',dict(status='sampled_remainder_integration_passed' if smoke else 'sampled_recovery_continuations_complete',
        searches=searches,additional_training_steps=sum(r['additional_steps'] for r in searches),
        new_steps_this_run=second['local_resume_steps'],elapsed_seconds=time.perf_counter()-started,
        shared_prefix_interactions=plan['shared_prefix_interactions'],interrupted_predecessor_preserved=True,
        final_opened=False,heldout_opened=False,promotion=False,scope=plan['quality_scope'],
        completed_at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-run',type=Path,default=Path('runs/league_sampled_ppo_finish_cpu_smoke_20261007'))
    args=parser.parse_args()
    try:run(args.source.resolve(),args.out.resolve(),args.smoke,args.smoke_run.resolve())
    except BaseException:
        io.write(args.out/'failure.json',dict(error=traceback.format_exc(),pid=os.getpid(),at=datetime.now(timezone.utc).isoformat()))
        raise
