"""Additional PPO learning with sampled-action development and grouped workers.

Preserves official FairFight and the original teacher. Two independent
continuations share one selected learner/optimizer prefix. Development uses
two fixed action replicas; neither the best action seed nor a fresh final test
chooses checkpoints. Existing failed greedy assessments remain unchanged.
"""
from argparse import ArgumentParser
from collections import defaultdict
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


def pair_rank(profiles):
    if len(profiles) != 2:
        raise ValueError('Exactly two fixed action replicas are required')
    return [sum(.5 * (p['mean_win_rate'] + p['group_balanced_win_rate']) for p in profiles) / 2,
            min(p['lower_quarter_score'] for p in profiles), min(p['worst_score'] for p in profiles),
            -max(p['losing_matchups'] for p in profiles)]


def should_extend(previous, latest, initial):
    return (latest[0] - previous[0] >= .005 - 1e-12 and latest[0] - initial[0] >= .005 - 1e-12
            and latest[1] >= previous[1] - 1e-12 and latest[2] >= previous[2] - 1e-12)


def evaluate(vec, spec, opponents, band, n, destination, reused=()):
    from tools import league_residual_train as base
    from tools.league_tournament_metrics import profile, code_groups
    if destination.exists():
        raise FileExistsError('Do not overwrite a completed development result')
    cached = {r['foe']: r for r in reused}
    if len(cached) != len(reused) or any(r['own'] != spec['id'] or r['band'] != band for r in reused):
        raise ValueError('Reused results do not match the same own policy and conditions')
    expected = {s['id'] for s in opponents}
    if not set(cached) <= expected:
        raise ValueError('Unexpected cached opponent')
    missing = [s for s in opponents if s['id'] not in cached]
    started = time.perf_counter()
    before = vec.env_method('worker_status')
    if missing:
        with base.isolated_rng():
            # One call dispatched to every physical process concurrently. The
            # generic adapter's arbitrary-index RPC loops groups serially;
            # long evaluation partitions must use this all-group dispatch.
            groups = vec.transport.env_method('inner_call', 'env_method', list(range(vec.group_size)),
                ('evaluate_partition', spec, missing, band, n), {})
        for group in groups:
            for partition in group:
                for record in partition:
                    if record['foe'] in cached:
                        raise ValueError('Duplicate development opponent')
                    cached[record['foe']] = record
    after = vec.env_method('worker_status')
    for a, b in zip(before, after, strict=True):
        for key in ('worker_index', 'episodes_started', 'opponent_draws', 'episode'):
            if a[key] != b[key]:
                raise AssertionError('Evaluation altered live training progression')
    rows = [cached[s['id']] for s in opponents]
    base.validate_records(rows, opponents, band, n)
    result = dict(request=dict(spec=spec, opponents=opponents, band=band, n=n), results=rows,
        profile=profile(rows, code_groups(opponents)), executed_new_games=len(missing) * n,
        reused_games=len(reused) * n, wall_seconds=time.perf_counter() - started,
        scope='Same fixed sampled-action replica, shared ICs and both seats. Paused training environments preserved.')
    io.write(destination, result)
    return result


def freeze(source, out, smoke, smoke_run):
    from tools.league_tournament_metrics import code_groups, group_weights
    if (out / 'plan.json').exists():
        raise FileExistsError('Use a fresh continuation directory')
    old = io.read(source / 'plan.json')
    io.verify(old)
    if io.read(source / 'completion.json')['status'] != 'sampled_development_complete':
        raise ValueError('Sampled checkpoint selection must be completed')
    choice = io.read(source / 'warm_start_choice.json')
    chosen = choice['chosen']
    if chosen != max(choice['candidates'], key=lambda r: (*r['rank'], -r['step'], -r['training_seed'])):
        raise ValueError('Warm-start choice disagrees with the frozen rule')
    previous = io.read(io.ROOT / old['source'] / 'plan.json')
    opponents = list(previous['opponents'])
    for candidate in old['candidates']:
        opponents.append(candidate['greedy'])
        opponents.extend(r['spec'] for r in candidate['replicas'])
    if len({s['id'] for s in opponents}) != len(opponents):
        raise ValueError('Duplicate training opponent IDs')
    early = next(c['greedy'] for c in old['candidates'] if c['training_seed'] == 3300 and c['step'] == 2097152)
    development = list(old['opponents']) + [r['spec'] for r in chosen['replicas']] + [early]
    if len({s['id'] for s in development}) != len(development):
        raise ValueError('Duplicate development opponent IDs')
    selected_ids = {r['spec']['id'] for r in chosen['replicas']}
    cached, cached_paths = defaultdict(list), []
    for path in sorted((source / 'matches').glob('match_*.json')):
        saved = io.read(path)
        if saved['job']['own']['id'] in selected_ids:
            if saved['job']['band'] != old['band'] or saved['job']['n'] != old['n']:
                raise ValueError('Changed baseline development conditions')
            cached[saved['job']['own']['id']].append(saved['result'])
            cached_paths.append(path)
    weakness = {}
    for foe in old['opponents']:
        scores = [next(r['summary']['score'] for r in cached[s] if r['foe'] == foe['id']) for s in selected_ids]
        weakness[foe['id']] = 1. - sum(scores) / len(scores)
    groups = code_groups(opponents)
    balanced, weak_total = group_weights(groups), sum(weakness.values())
    if weak_total <= 0:
        raise ValueError('No observed development weaknesses')
    probabilities = [.4 / len(opponents) + .3 * balanced[s['id']] + .3 * weakness.get(s['id'], 0.) / weak_total for s in opponents]
    if smoke:
        development = [s for s in development if s['id'] in ('ace', 'ddqn_s0', 'temporal_extend_right', chosen['replicas'][0]['spec']['id'])]
        cached = {}
    source_hashes, input_hashes = dict(old['source_sha256']), dict(old['input_sha256'])
    for name in ('tools/league_sampled_ppo_continue.py', 'tests/test_league_sampled_ppo_continue.py',
                 'experiments/league/grouped_vec_env.py', 'tools/league_grouped_ppo_benchmark.py',
                 'tools/league_residual_profile.py', 'tools/league_residual_recover.py'):
        source_hashes[name] = io.sha(io.ROOT / name)
    paths = [source / 'plan.json', source / 'completion.json', source / 'warm_start_choice.json',
             io.ROOT / chosen['learner'], *cached_paths,
             io.ROOT / 'runs/league_grouped_ppo_benchmark_20261007/completion.json',
             io.ROOT / 'runs/league_grouped_ppo_benchmark_20261007/analysis.json',
             io.ROOT / 'runs/sampled_continue_software_checks_20261007.xml']
    if not smoke:
        check = io.read(smoke_run / 'completion.json')
        tested = io.read(smoke_run / 'plan.json')
        if check['status'] != 'sampled_continuation_integration_passed' or any(
                tested['source_sha256'].get(name) != digest for name, digest in source_hashes.items()):
            raise ValueError('Require matching completed integration run')
        paths += [smoke_run / 'plan.json', smoke_run / 'completion.json']
    for path in paths:
        input_hashes[io.relative(path)] = io.sha(path)
    plan = dict(previous=io.relative(source), teacher=previous['teacher'], warm_start=chosen,
        source_checkpoint_steps=chosen['step'], shared_prefix_interactions=previous['shared_prefix_steps'] + chosen['step'],
        early_greedy_comparator=early, opponents=opponents, groups=groups, probabilities=probabilities,
        mixture='CoverageSampler retains alternate full shuffled archive coverage; weighted half is40% uniform,30% source-group balance,30% fixed weakness from both initial action replicas. All72 previous foes plus8 greedy and16 sampled PPO snapshots retained.',
        weakness=weakness, development_opponents=development, development_band=170020000 if smoke else old['band'],
        development_n=2 if smoke else old['n'], action_replicas=[4900, 4901], cached_baseline=dict(cached),
        processes=4, workers=16, ppo=dict(previous['ppo'], n_steps=256), seeds=[4599] if smoke else [4500, 4501],
        training_bands=[170010000] if smoke else [260000000, 261000000],
        chunk_steps=32768 if smoke else 1048576, base_chunks=1 if smoke else 2, maximum_chunks=1 if smoke else 4,
        budget_rule='After2chunks(2,097,152 new steps/seed), extend to4chunks only if retained two-replica rank first component improves>=.005 since chunk1 and exceeds initial by>=.005, without reduced retained lower-quarter or worst score. No final outcomes used.',
        quality_scope='Two new continuation RNGs after one selected actor/critic/optimizer prefix, not independent from-scratch replications. Development deliberately reused for selection. No final/heldout opened and no policy promotion by this learner.',
        resume_scope='Learner/optimizer checkpoints and RNG retained; JSBSim mid-episode state is not serialized. Any recovery must declare fresh streams and lost partial rollout.',
        smoke=smoke, source_sha256=source_hashes, input_sha256=input_hashes, environment=io.environment())
    if not smoke:
        for band in plan['training_bands']:
            reservation = io.ROOT / f'runs/training_reservation_{band}.json'
            if reservation.exists():
                raise FileExistsError('Training band already reserved')
            io.write(reservation, dict(run=io.relative(out), start=band, stop_exclusive=band + 1000000))
    io.verify(plan)
    io.write(out / 'plan.json', plan)
    for name in source_hashes:
        target = out / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(io.ROOT / name, target)
    return plan


def same_tree(a, b):
    import torch
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return a == b


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
        progress = base.Progress(folder, seed)
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
    io.write(out / 'runtime.json', dict(pid=os.getpid(), processes=4, virtual_environments=16,
        started_at=datetime.now(timezone.utc).isoformat(), device='cpu', smoke=smoke))
    started, searches, baseline = time.perf_counter(), [], None
    for ordinal, seed in enumerate(plan['seeds']):
        gc.collect()
        io.verify(plan)
        result, baseline = train_seed(out, plan, seed, ordinal, baseline)
        searches.append(result)
    io.verify(plan)
    io.write(out / 'completion.json', dict(status='sampled_continuation_integration_passed' if smoke else 'sampled_continuations_complete',
        searches=searches, additional_training_steps=sum(r['additional_steps'] for r in searches),
        shared_prefix_interactions=plan['shared_prefix_interactions'], elapsed_seconds=time.perf_counter() - started,
        final_opened=False, heldout_opened=False, promotion=False, completed_at=datetime.now(timezone.utc).isoformat(),
        next_action='Analyze retained sampled development gains before a separately frozen fresh comparison against teacher and preserved2M greedy policy.', scope=plan['quality_scope']))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-run', type=Path, default=Path('runs/league_sampled_ppo_smoke_20261007'))
    args = parser.parse_args()
    try:
        run(args.source.resolve(), args.out.resolve(), args.smoke, args.smoke_run.resolve())
    except BaseException:
        io.write(args.out / 'failure.json', dict(error=traceback.format_exc(), pid=os.getpid(), at=datetime.now(timezone.utc).isoformat()))
        raise
