"""Matched PPO continuations: formula control, balanced coverage, lower prior.

All arms share actor/critic/optimizer tensors and official physics; only the
frozen opponent selection treatment or fixed actor logit prior differs.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from functools import partial
from io import BytesIO
from pathlib import Path
import gc,json,os,shutil,time,traceback,zipfile
import numpy as np
from tools import league_thread_benchmark as io
from tools.league_sampled_ppo_continue import evaluate,pair_rank,should_extend,same_tree


def make_progress(folder,seed):
    from tools import league_residual_train as base
    from experiments.league.memory_pause import guarded_step,wait_for_headroom
    class Progress(base.Progress):
        def _on_step(self):return guarded_step(super()._on_step,self.pause)
        def pause(self):
            def emit(row):
                with (folder/'resource_pauses.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(row,steps=self.model.num_timesteps))+'\n')
                print('strategy resource pause',row['stage'],row['elapsed_seconds'],flush=True)
            wait_for_headroom(io.resources,emit)
    return Progress(folder,seed)


def freeze(assessment,out,smoke,smoke_run):
    if out.exists():raise FileExistsError('Use a new matched strategy directory')
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Set thread limits first')
    if io.process_live(io.read(assessment/'runtime.json')['pid']):raise RuntimeError('Assessment still live')
    if io.read(assessment/'completion.json')['status']!='sampled_fresh_assessment_complete':raise ValueError('Complete fresh assessment required')
    assessed=io.read(assessment/'plan.json');io.verify(assessed)
    source=io.ROOT/assessed['source'];old=io.read(source/'plan.json');chosen=assessed['chosen']
    checks_path=io.ROOT/'runs/league_prior_adjustment_checks_20261007/completion.json';checks=io.read(checks_path)
    if checks['status']!='passed' or checks['source']!=chosen['learner'] or io.sha(io.ROOT/checks['source'])!=checks['source_sha256'] or io.sha(io.ROOT/checks['prototype_checkpoint'])!=checks['prototype_sha256']:raise ValueError('Mismatched prior adjustment checks')
    import xml.etree.ElementTree as ET
    xml=io.ROOT/'runs/periodic_coverage_checks_20261007.xml'
    if any(int(s.get('failures','0')) or int(s.get('errors','0')) for s in ET.parse(xml).iter('testsuite')):raise ValueError('Coverage tests failed')
    opponents=list(assessed['opponents'])
    for replica in chosen['replicas']:
        spec=replica['spec'];prior=next((s for s in opponents if s['id']==spec['id']),None)
        if prior is not None and prior!=spec:raise ValueError('Opponent ID collision')
        if prior is None:opponents.append(spec)
    analysis=io.read(assessment/'analysis.json');scores=analysis['roles']['candidate']['opponents']
    from tools.league_tournament_metrics import code_groups,group_weights
    from tools import league_residual_train as base
    groups=code_groups(opponents);balanced=group_weights(groups)
    weak={s['id']:.05+1-scores.get(s['id'],dict(wins=.5,games=1))['wins']/scores.get(s['id'],dict(wins=.5,games=1))['games'] for s in opponents}
    total=sum(weak.values());weak={k:v/total for k,v in weak.items()}
    mixtures={'control':(.4,.3,.3),'balanced':(.2,.4,.4),'lower_prior':(.4,.3,.3)}
    arms=[dict(name=name,prior_bias=3 if name=='lower_prior' else 6,coverage_period=4 if name=='balanced' else 2,
        raw_mixture=list(weights),probabilities=[weights[0]/len(opponents)+weights[1]*balanced[s['id']]+weights[2]*weak[s['id']] for s in opponents]) for name,weights in mixtures.items()]
    temporal=[s['id'] for s in opponents if s['id'].startswith('temporal_')]
    development=base.development_panel(opponents,groups,temporal,old['teacher']['id'],size=32)
    if smoke:development=[next(s for s in opponents if s['id']==name) for name in ('ace','ddqn_s0','temporal_extend_right','evader')]
    sources=dict(assessed['source_sha256']);inputs=dict(assessed['input_sha256'])
    for name in ('tools/league_sampled_strategy.py','experiments/league/strategy_worker.py','experiments/league/periodic_coverage.py','tests/test_league_periodic_coverage.py','experiments/league/memory_pause.py'):
        sources[name]=io.sha(io.ROOT/name)
    if not smoke:
        tested=io.read(smoke_run/'plan.json');done=io.read(smoke_run/'completion.json')
        if done['status']!='matched_strategy_integration_passed' or tested['source_sha256']!=sources:raise ValueError('Matching completed integration required')
    from stable_baselines3 import PPO
    from tools.league_residual_action_mode import build_variant
    model=PPO.load(io.ROOT/checks['prototype_checkpoint'],device='cpu')
    greedy=base.save_checkpoint(model,out/'models/initial_bias3',old['teacher'],out.name+'_initial_bias3')
    warm3=dict(learner=checks['prototype_checkpoint'],greedy=greedy,replicas=[dict(action_seed=k,spec=build_variant(greedy,out/f'models/initial_bias3_sampled_{k}',k,out.name+f'_initial_bias3_a{k}')) for k in (4900,4901)])
    del model;gc.collect()
    warm6={k:chosen[k] for k in ('learner','greedy','replicas')}
    plan=dict(old,smoke=smoke,assessment=io.relative(assessment),opponents=opponents,groups=groups,weakness=weak,
        arms=arms,warm_starts={'6':warm6,'3':warm3},warm_start=warm6,source_checkpoint_steps=chosen['step'],
        shared_prefix_interactions=old['shared_prefix_interactions']+chosen['step'],
        seeds=[5399] if smoke else [5300,5301],training_bands=[170110000] if smoke else [270000000,271000000],
        processes=2 if smoke else 8,workers=8 if smoke else 32,ppo=dict(old['ppo'],n_steps=512 if smoke else 128),
        development_opponents=development,development_band=170120000 if smoke else 82000000,development_n=2 if smoke else 4,
        chunk_steps=32768 if smoke else 1048576,base_chunks=1,maximum_chunks=1,cached_baseline={},
        source_sha256=sources,input_sha256=inputs,environment=io.environment(),
        budget_rule='Exactly one matched1,048,576-step pilot per arm/RNG; no single-arm automatic extension. Analyze both paired RNGs before matched additional budget.',
        quality_scope='Three treatments on the same updated100-foe archive, two paired learner RNGs after one pretrained prefix. Control retains70/15/15 effective episode mixture; balanced40/30/30; lower_prior only changes fixed bias6to3. Same sampled evaluation; no from-scratch, final, holdout, or promotion claim.',
        resume_scope='Each arm restores the shared actor/critic/optimizer tensors with new deterministic learner/episode streams. Bias3 metadata is the declared actor offset treatment. Common prefix counted once.',
        experiment_scope='Assessment81M is consumed development for this new curriculum. New82M sparse development chooses checkpoints; no unused final/holdout opened. Matched arms intentionally share training IC bands within each RNG.',
        probabilities=arms[0]['probabilities'],coverage_period=2)
    extra=[assessment/'plan.json',assessment/'completion.json',assessment/'analysis.json',checks_path,io.ROOT/checks['prototype_checkpoint'],xml]
    if not smoke:extra += [smoke_run/'plan.json',smoke_run/'completion.json']
    for spec in [s for warm in (warm6,warm3) for s in [warm['greedy']]+[r['spec'] for r in warm['replicas']]]+opponents:
        if spec['kind']=='submission':extra+=list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]
    for path in extra:inputs[io.relative(path)]=io.sha(path)
    if not smoke:
        for band in plan['training_bands']:
            reservation=io.ROOT/f'runs/training_reservation_{band}.json'
            if reservation.exists():raise FileExistsError('Training band reserved')
            io.write(reservation,dict(run=io.relative(out),start=band,stop_exclusive=band+1000000,shared_by_matched_arms=[a['name'] for a in arms]))
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,target)
    return plan


def train_seed(out, plan, seed, ordinal, baseline):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from experiments.league.grouped_vec_env import GroupedSubprocVecEnv
    from tools import league_residual_train as base
    from tools.league_residual_profile import tensor_digest
    from experiments.league.strategy_worker import make_strategy_worker as make_profile_worker
    from tools.league_residual_recover import load_continuation
    from tools.league_residual_action_mode import build_variant
    folder = out / f's{seed}'
    folder.mkdir()
    local = dict(plan, seed=seed, ic_band=plan['training_bands'][ordinal])
    factories = [partial(make_profile_worker, local, i, False) for i in range(plan['workers'])]
    vec, model = None, None
    try:
        if min(io.resources()[k] for k in ('commit_headroom_gib', 'available_memory_gib')) < 4.5:
            raise MemoryError('Insufficient grouped learner startup headroom')
        vec = GroupedSubprocVecEnv(factories, processes=plan['processes'])
        statuses = vec.env_method('profile_status')
        if any(s['cached_opponents'] != len(plan['opponents']) for s in statuses):
            raise AssertionError('Incomplete training opponent cache')
        io.write(folder / 'initialized_workers.json', dict(virtual_environments=statuses,
                 physical_workers=len({s['memory']['pid'] for s in statuses}), resources=io.resources()))
        if io.resources()['commit_headroom_gib'] < 2.8:
            raise MemoryError('Insufficient initialized learner headroom')
        model, prefix = load_continuation(io.ROOT / plan['warm_start']['learner'], vec, seed, plan['ppo']['n_steps'])
        if model.policy.prior_bias != plan['prior_bias']:
            raise AssertionError('Loaded actor prior differs from declared treatment')
        with zipfile.ZipFile(io.ROOT / plan['warm_start']['learner']) as archive:
            expected_policy = torch.load(BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)
            expected_optimizer = torch.load(BytesIO(archive.read('policy.optimizer.pth')), map_location='cpu', weights_only=True)
        if prefix != plan['source_checkpoint_steps'] or not same_tree(model.policy.state_dict(), expected_policy) or not same_tree(model.policy.optimizer.state_dict(), expected_optimizer):
            raise AssertionError('Warm actor/critic/optimizer changed while loading')
        initial_digest = tensor_digest(model.policy.state_dict())
        del expected_policy, expected_optimizer
        model.set_logger(configure(str(folder / 'metrics'), ['csv']))
        if baseline is None:
            records = [evaluate(vec, r['spec'], plan['development_opponents'], plan['development_band'], plan['development_n'],
                out / f"baseline_action_{r['action_seed']}.json", plan['cached_baseline'].get(r['spec']['id'], []))
                for r in plan['warm_start']['replicas']]
            baseline = dict(learner=plan['warm_start']['learner'], greedy=plan['warm_start']['greedy'],
                replicas=plan['warm_start']['replicas'], rank=pair_rank([r['profile'] for r in records]), step=0)
            io.write(out / 'baseline.json', dict(**baseline, new_games=sum(r['executed_new_games'] for r in records), reused_games=sum(r['reused_games'] for r in records)))
        best = baseline
        progress = make_progress(folder, seed)
        history, limit, extended = [], plan['base_chunks'], False
        for block in range(plan['maximum_chunks']):
            if block >= limit: break
            started = time.perf_counter()
            model.learn(plan['chunk_steps'], callback=progress, reset_num_timesteps=False)
            learning_seconds = time.perf_counter() - started
            checkpoint = folder / f'checkpoints/step_{model.num_timesteps}'
            identity = f'{out.name}_s{seed}_t{model.num_timesteps}'
            greedy = base.save_checkpoint(model, checkpoint, plan['teacher'], identity)
            replicas, records = [], []
            for action_seed in plan['action_replicas']:
                spec = build_variant(greedy, checkpoint / f'sampled_{action_seed}', action_seed, identity + f'_a{action_seed}')
                replicas.append(dict(action_seed=action_seed, spec=spec))
                records.append(evaluate(vec, spec, plan['development_opponents'], plan['development_band'], plan['development_n'], checkpoint / f'development_{action_seed}.json'))
            candidate = dict(learner=io.relative(checkpoint / 'learner.zip'), greedy=greedy, replicas=replicas,
                rank=pair_rank([r['profile'] for r in records]), step=model.num_timesteps)
            if tuple(candidate['rank']) > tuple(best['rank']): best = candidate
            row = dict(chunk=block + 1, steps=model.num_timesteps, learning_seconds=learning_seconds,
                evaluation_seconds=sum(r['wall_seconds'] for r in records), evaluation_games=sum(r['executed_new_games'] for r in records),
                candidate=candidate, retained=best, episodes=progress.completed,
                training_outcomes=progress.counts.copy(), opponent_episode_counts=progress.foes.copy())
            history.append(row)
            io.write(folder / f'chunk_{block + 1:02d}.json', row)
            if block + 1 == plan['base_chunks']:
                extended = not plan['smoke'] and should_extend(history[0]['retained']['rank'], best['rank'], baseline['rank'])
                if extended: limit = plan['maximum_chunks']
                io.write(folder / 'budget_decision.json', dict(extend=extended, chunks=limit, baseline_rank=baseline['rank'],
                    first_chunk_rank=history[0]['retained']['rank'], latest_rank=best['rank'], rule=plan['budget_rule']))
            progress.report('sampled_checkpoint_complete')
        result = dict(seed=seed, selected=best, history=history, additional_steps=model.num_timesteps,
            training_episodes=progress.completed, outcomes=progress.counts, opponent_episode_counts=progress.foes,
            initial_parameters_sha256=initial_digest, final_parameters_sha256=tensor_digest(model.policy.state_dict()),
            minimum_commit_headroom_gib=progress.minimum_commit, extended=extended)
        if plan['smoke']:
            restored = PPO.load(checkpoint / 'learner.zip', device='cpu')
            if not same_tree(restored.policy.state_dict(), model.policy.state_dict()) or not same_tree(restored.policy.optimizer.state_dict(), model.policy.optimizer.state_dict()):
                raise AssertionError('Saved learner/optimizer did not restore exactly')
            if initial_digest == result['final_parameters_sha256']:
                raise AssertionError('Integration run did not update weights')
            result['learner_optimizer_restore_exact'] = True
        io.write(folder / 'result.json', result)
        return result, baseline
    except BaseException:
        if model is not None:
            model.save(folder / 'interrupted_learner.zip')
            io.write(folder / 'interrupted_checkpoint.json', dict(additional_steps=model.num_timesteps, scope=plan['resume_scope']))
        raise
    finally:
        if vec is not None: vec.close()


def run(assessment, out, smoke, smoke_run):
    import torch
    torch.set_num_threads(1)
    if torch.__version__ != '2.14.1+cpu':
        raise ValueError('Use the qualified CPU environment')
    started = time.perf_counter()
    plan = freeze(assessment, out, smoke, smoke_run)
    io.write(out/'runtime.json', dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(), resources=io.resources()))
    baselines, results = {}, []
    try:
        for ordinal, seed in enumerate(plan['seeds']):
            arms = plan['arms'] if ordinal == 0 else list(reversed(plan['arms']))
            for arm in arms:
                io.verify(plan)
                gc.collect()
                prior = str(arm['prior_bias'])
                local = dict(plan, probabilities=arm['probabilities'], coverage_period=arm['coverage_period'],
                             prior_bias=arm['prior_bias'], warm_start=plan['warm_starts'][prior])
                folder = out/arm['name']
                folder.mkdir(exist_ok=True)
                print('strategy learning start', arm['name'], seed, plan['chunk_steps'], flush=True)
                result, baseline = train_seed(folder, local, seed, ordinal, baselines.get(prior))
                baselines[prior] = baseline
                results.append(dict(result, arm=arm['name'], prior_bias=arm['prior_bias'], coverage_period=arm['coverage_period']))
                io.write(out/'completed_searches.json', dict(searches=results))
        expected = io.read(io.ROOT/'runs/league_prior_adjustment_checks_20261007/completion.json')['actor_critic_tensor_sha256']
        if any(r['initial_parameters_sha256'] != expected for r in results):
            raise AssertionError('Matched arms did not start with identical actor/critic tensors')
        io.verify(plan)
        io.write(out/'completion.json', dict(status='matched_strategy_integration_passed' if smoke else 'matched_strategy_pilots_complete',
            searches=results, additional_training_steps=sum(r['additional_steps'] for r in results),
            shared_prefix_interactions=plan['shared_prefix_interactions'], elapsed_seconds=time.perf_counter()-started,
            quality_scope=plan['quality_scope'], final_test_opened=False, heldout_opened=False, promoted=False))
        print('matched strategy complete', out, flush=True)
    except BaseException:
        io.write(out/'failure.json', dict(at=datetime.now(timezone.utc).isoformat(), traceback=traceback.format_exc(), resources=io.resources()))
        raise


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--assessment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-run', type=Path, default=Path('runs/league_sampled_strategy_smoke_20261007'))
    args = parser.parse_args()
    run(args.assessment.resolve(), args.out.resolve(), args.smoke, args.smoke_run.resolve())

