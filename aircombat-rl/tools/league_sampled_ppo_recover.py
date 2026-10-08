"""Two-process recovery from the preserved sampled-PPO emergency checkpoint.

The stopped run is immutable. Start two fresh continuation RNGs/IC streams
after its shared emergency prefix, discard its16 untrained transitions, and
retain the previously better8M policy as a development fallback. No OS service
or memory safety-floor changes. Full rollout size stays4096 at8x512.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from functools import partial
from io import BytesIO
import gc
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import zipfile
from tools import league_thread_benchmark as io
from tools.league_sampled_ppo_continue import pair_rank, should_extend, evaluate, same_tree


def freeze(source, out, smoke, smoke_run):
    import torch
    from stable_baselines3 import PPO
    from tools.league_residual_export import export
    from tools.league_residual_action_mode import build_variant
    if (out / 'plan.json').exists():
        raise FileExistsError('Use a fresh recovery directory')
    old = io.read(source / 'plan.json')
    io.verify(old)
    failure = io.read(source / 'failure.json')
    stopped = io.read(source / 'termination_analysis.json')
    if 'MemoryError: Commit/physical headroom below safe operating floor' not in failure['error'] or stopped['exit_code'] != 1:
        raise ValueError('Expected verified resource-guard termination')
    if io.process_live(stopped['controller_pid']):
        raise RuntimeError('Stopped controller PID is live; inspect identity before recovery')
    saved = source / 's4500/interrupted_learner.zip'
    if io.sha(saved) != stopped['checkpoint_sha256']:
        raise ValueError('Emergency checkpoint changed')
    model = PPO.load(saved, device='cpu')
    if model.num_timesteps != stopped['saved_interactions']:
        raise ValueError('Emergency learner step mismatch')
    greedy = export(out / 'initial_submission', old['teacher'], model.policy, out.name + '_initial')
    del model
    gc.collect()
    replicas = [dict(action_seed=seed, spec=build_variant(greedy, out / f'initial_sampled_{seed}', seed,
                out.name + f'_initial_a{seed}')) for seed in old['action_replicas']]
    plan = dict(old)
    plan['source_sha256'] = dict(old['source_sha256'])
    plan['input_sha256'] = dict(old['input_sha256'])
    for name in ('tools/league_sampled_ppo_recover.py', 'tests/test_league_sampled_ppo_recover.py'):
        plan['source_sha256'][name] = io.sha(io.ROOT / name)
    plan.update(previous=io.relative(source), warm_start=dict(learner=io.relative(saved), greedy=greedy, replicas=replicas),
        source_checkpoint_steps=stopped['saved_interactions'],
        shared_prefix_interactions=old['shared_prefix_interactions'] + stopped['saved_interactions'],
        discarded_interactions=stopped['pending_rollout_interactions_discarded_on_fresh_resume'],
        processes=2, workers=8, ppo=dict(old['ppo'], n_steps=512), startup_headroom_gib=3.7,
        seeds=[4699] if smoke else [4600, 4601], training_bands=[170040000] if smoke else [262000000, 263000000],
        environment_seed_rule='After learner RNG reset, explicitly set virtual environment seeds to1000*learner_seed+i, avoiding shared opponent-draw streams between the two continuations. Official IC bands are also disjoint.',
        cached_baseline={}, fallback_baseline=None if smoke else io.read(source / 'baseline.json'),
        development_band=170050000 if smoke else old['development_band'], development_n=2 if smoke else old['development_n'],
        chunk_steps=32768 if smoke else old['chunk_steps'], base_chunks=1 if smoke else old['base_chunks'],
        maximum_chunks=1 if smoke else old['maximum_chunks'], smoke=smoke, environment=io.environment(),
        quality_scope='Two fresh learner RNGs after the same shared emergency actor/critic/optimizer prefix. Fresh episode/ opponent streams and different8-env topology; not exact interrupted-trajectory continuation or from-scratch replication. Preserve earlier8M development fallback. No final/heldout or promotion.',
        recovery_scope='Previous4500 run ended at1,196,048 interactions, with1,196,032 in completed rollouts and16 pending transitions discarded. Previous4501 never started. Both new learners start at the saved weights/optimizer; emergency RNG/mid-episode simulator state is not restored.',
        memory_rule='Two physical workers allow a3.7GiB pre-pool headroom threshold; initialized headroom must still exceed2.8GiB. Running guard stays1.8GiB commit/1.5GiB physical. Record per-process memory with normal telemetry; stop and save if guard triggers.')
    if smoke:
        wanted = {'ace', 'ddqn_s0', 'temporal_extend_right', old['warm_start']['replicas'][0]['spec']['id']}
        plan['development_opponents'] = [s for s in old['development_opponents'] if s['id'] in wanted]
    paths = [source / n for n in ('plan.json', 'failure.json', 'termination_analysis.json', 'baseline.json',
             'baseline_action_4900.json', 'baseline_action_4901.json', 's4500/interrupted_checkpoint.json')]
    paths += [saved, io.ROOT / 'runs/sampled_recovery_software_checks_20261007.xml']
    for spec in [greedy] + [r['spec'] for r in replicas]:
        paths.extend((io.ROOT / spec['design']).glob('*'))
    if not smoke:
        checked = io.read(smoke_run / 'plan.json')
        if io.read(smoke_run / 'completion.json')['status'] != 'sampled_recovery_integration_passed' or any(
                checked['source_sha256'].get(name) != digest for name, digest in plan['source_sha256'].items()):
            raise ValueError('Matching completed recovery integration run required')
        paths += [smoke_run / 'plan.json', smoke_run / 'completion.json']
    for path in paths:
        if path.is_file(): plan['input_sha256'][io.relative(path)] = io.sha(path)
    if not smoke:
        for band in plan['training_bands']:
            reservation = io.ROOT / f'runs/training_reservation_{band}.json'
            if reservation.exists(): raise FileExistsError('Training band already reserved')
            io.write(reservation, dict(run=io.relative(out), start=band, stop_exclusive=band + 1000000))
    io.verify(plan)
    io.write(out / 'plan.json', plan)
    for name in plan['source_sha256']:
        target = out / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(io.ROOT / name, target)
    return plan

# train_seed and run below are adapted from the frozen prior controller;
# its original source and interrupted results are never edited.


def choose_baseline(initial, fallback):
    if fallback is not None and tuple(fallback['rank']) > tuple(initial['rank']):
        return fallback
    return initial


def train_seed(out, plan, seed, ordinal, baseline):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from experiments.league.grouped_vec_env import GroupedSubprocVecEnv
    from tools import league_residual_train as base
    from tools.league_residual_profile import make_profile_worker, tensor_digest
    from tools.league_residual_recover import load_continuation
    from tools.league_residual_action_mode import build_variant
    folder = out / f's{seed}'
    folder.mkdir()
    local = dict(plan, seed=seed, ic_band=plan['training_bands'][ordinal])
    factories = [partial(make_profile_worker, local, i, False) for i in range(plan['workers'])]
    vec, model = None, None
    try:
        if min(io.resources()[k] for k in ('commit_headroom_gib', 'available_memory_gib')) < plan['startup_headroom_gib']:
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
        vec.seed(seed * 1000)
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
        baseline = choose_baseline(baseline, plan.get('fallback_baseline'))
        io.write(folder / 'retained_starting_baseline.json', baseline)
        best = baseline
        class WatchedProgress(base.Progress):
            def report(self, stage):
                super().report(stage)
                statuses = vec.env_method('profile_status')
                workers = {s['memory']['pid']: s['memory'] for s in statuses}
                row = dict(at=datetime.now(timezone.utc).isoformat(), steps=self.model.num_timesteps,
                           parent=io.own_memory(), workers=list(workers.values()), resources=io.resources())
                with (folder / 'process_memory.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row) + '\n')

            def _on_step(self):
                try:
                    return super()._on_step()
                except MemoryError:
                    io.write(folder / 'guard_failure_context.json', dict(steps=self.model.num_timesteps,
                        minimum_commit_headroom_gib=self.minimum_commit, sample_after_exception=io.resources(),
                        parent=io.own_memory(), scope='Minimum commit includes the triggering check; fresh resource snapshot is taken immediately after the exception.'))
                    raise

        progress = WatchedProgress(folder, seed)
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

def run(source, out, smoke, smoke_run):
    if any(os.environ.get(k) != '1' for k in io.THREAD_KEYS):
        raise ValueError('Set thread limits before Python starts')
    plan = freeze(source, out, smoke, smoke_run)
    import torch
    torch.set_num_threads(1)
    io.write(out / 'runtime.json', dict(pid=os.getpid(), processes=plan['processes'], virtual_environments=plan['workers'],
        started_at=datetime.now(timezone.utc).isoformat(), device='cpu', smoke=smoke))
    started, searches, baseline = time.perf_counter(), [], None
    for ordinal, seed in enumerate(plan['seeds']):
        gc.collect()
        io.verify(plan)
        result, baseline = train_seed(out, plan, seed, ordinal, baseline)
        searches.append(result)
    io.verify(plan)
    io.write(out / 'completion.json', dict(status='sampled_recovery_integration_passed' if smoke else 'sampled_recovery_continuations_complete',
        searches=searches, additional_training_steps=sum(r['additional_steps'] for r in searches),
        shared_prefix_interactions=plan['shared_prefix_interactions'], elapsed_seconds=time.perf_counter() - started,
        final_opened=False, heldout_opened=False, promotion=False, completed_at=datetime.now(timezone.utc).isoformat(),
        next_action='Analyze retained sampled development gains before a separately frozen fresh comparison against teacher and preserved2M greedy policy.', scope=plan['quality_scope']))

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-run', type=Path, default=Path('runs/league_sampled_ppo_recovery_smoke_20261007'))
    args = parser.parse_args()
    try:
        run(args.source.resolve(), args.out.resolve(), args.smoke, args.smoke_run.resolve())
    except BaseException:
        io.write(args.out / 'failure.json', dict(error=traceback.format_exc(), pid=os.getpid(), at=datetime.now(timezone.utc).isoformat()))
        raise
