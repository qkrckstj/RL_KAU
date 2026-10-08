"""Qualify grouped real-flight PPO, then measure four-process throughput.

The same four virtual environments must reproduce the completed ordinary
profile exactly. ABBA then compares 4 versus 16 virtual environments with the
same 4096-transition PPO batch. Those topologies use different rollout data;
this is an execution configuration study, not a policy-quality comparison.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from functools import partial
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import median
import time
import traceback
from tools import league_thread_benchmark as io


def freeze(source, out):
    if (out / 'plan.json').exists():
        raise FileExistsError('Use a new benchmark directory')
    if any(os.environ.get(k) != '1' for k in io.THREAD_KEYS):
        raise ValueError('Set thread limits before imports')
    old = io.read(source / 'queue_plan.json')
    io.verify(old)
    if io.read(source / 'completion.json')['status'] != 'profiling_complete':
        raise ValueError('Require completed ordinary profile')
    if not io.read(source / 'analysis.json')['trajectories_and_final_parameters_equal']:
        raise ValueError('Original timing qualification failed')
    plan = dict(old)
    plan['source_sha256'] = dict(old['source_sha256'])
    plan['input_sha256'] = dict(old['input_sha256'])
    for name in ('experiments/league/grouped_vec_env.py', 'tests/test_league_grouped_vec_env.py',
                 'tools/league_grouped_ppo_benchmark.py'):
        plan['source_sha256'][name] = io.sha(io.ROOT / name)
    for path in [source / n for n in ('queue_plan.json', 'completion.json', 'analysis.json', 'unmeasured_a/result.json')] + [io.ROOT / 'runs/grouped_vec_software_checks_v2_20261007.xml']:
        plan['input_sha256'][io.relative(path)] = io.sha(path)
    plan.update(previous_profile=io.relative(source), environment=io.environment(),
        qualification=dict(id='qualification_grouped2_env4', grouped=True, processes=2, envs=4),
        trials=[dict(id='ordinary4_a', grouped=False, processes=4, envs=4),
                dict(id='grouped4_env16_b', grouped=True, processes=4, envs=16),
                dict(id='grouped4_env16_c', grouped=True, processes=4, envs=16),
                dict(id='ordinary4_d', grouped=False, processes=4, envs=4)],
        rollout_batch=4096, steps_per_trial=65536,
        adoption_rule='Exact real-learning replay at4 virtual environments, deterministic repeats within each topology, grouped16 faster in both ABBA pairs and median time >=10% lower, minimum measured commit headroom>=2.8GiB and physical headroom>=2GiB.',
        scope='Compatibility and execution timing only, reused compatibility ICs. All72 opponent actors cached in every environment. Four processes in timing trials; virtual environment count4 versus16 changes rollout composition and per-env GAE horizon1024 versus256. No policy-quality promotion or GitHub publication.')
    io.verify(plan)
    io.write(out / 'plan.json', plan)
    for name in plan['source_sha256']:
        destination = out / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(io.ROOT / name, destination)
    return plan


def trial(plan, setting, out):
    import numpy as np
    import torch
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from experiments.league.grouped_vec_env import GroupedSubprocVecEnv
    from tools.league_residual_profile import make_profile_worker, tensor_digest
    from tools.league_residual_recover import load_continuation
    from tools.league_residual_train import Progress
    torch.set_num_threads(1)
    folder = out / setting['id']
    folder.mkdir()
    if min(io.resources()[k] for k in ('commit_headroom_gib', 'available_memory_gib')) < 4.5:
        raise MemoryError('Insufficient benchmark startup headroom')
    local = dict(plan, workers=setting['envs'])
    factories = [partial(make_profile_worker, local, i, False) for i in range(setting['envs'])]
    vec = None
    with io.Sampler() as sampler:
        try:
            started = time.perf_counter()
            vec = (GroupedSubprocVecEnv(factories, processes=setting['processes']) if setting['grouped']
                   else SubprocVecEnv(factories, start_method='spawn'))
            statuses = vec.env_method('profile_status')
            if any(s['cached_opponents'] != len(plan['opponents']) for s in statuses):
                raise AssertionError('Incomplete opponent cache')
            if io.resources()['commit_headroom_gib'] < 2.8:
                raise MemoryError('Insufficient initialized benchmark headroom')
            n_steps = plan['rollout_batch'] // setting['envs']
            model, prefix = load_continuation(io.ROOT / plan['checkpoint'], vec, plan['seed'], n_steps)
            if prefix != plan['checkpoint_steps']:
                raise AssertionError('Wrong starting checkpoint')
            initial = tensor_digest(model.policy.state_dict())
            model.set_logger(configure(str(folder / 'metrics'), ['csv']))

            class Trace(Progress):
                def __init__(self):
                    super().__init__(folder, plan['seed'])
                    self.trace = hashlib.sha256()
                    self.hash_seconds = 0.

                def _on_step(self):
                    start = time.perf_counter()
                    for name in ('actions', 'new_obs', 'rewards', 'dones'):
                        self.trace.update(np.ascontiguousarray(self.locals[name]).tobytes())
                    for info in self.locals['infos']:
                        keys = ('outcome', 'won', 'own_health', 'opp_health', 't', 'official_episode_flags', 'league_episode')
                        self.trace.update(json.dumps({k: info[k] for k in keys if k in info}, sort_keys=True).encode())
                    self.hash_seconds += time.perf_counter() - start
                    return super()._on_step()

            callback = Trace()
            startup = time.perf_counter() - started
            started, cpu_started = time.perf_counter(), time.process_time()
            model.learn(plan['steps_per_trial'], callback=callback, reset_num_timesteps=False)
            elapsed, cpu_seconds = time.perf_counter() - started, time.process_time() - cpu_started
            statuses = vec.env_method('profile_status')
            memories = {s['memory']['pid']: s['memory'] for s in statuses}
            if len(memories) != setting['processes']:
                raise AssertionError('Wrong physical worker count')
            result = dict(setting=setting, steps=model.num_timesteps, n_steps=n_steps,
                startup_seconds=startup, learning_seconds=elapsed, steps_per_second=model.num_timesteps / elapsed,
                initial_parameters_sha256=initial, final_parameters_sha256=tensor_digest(model.policy.state_dict()),
                trajectory_sha256=callback.trace.hexdigest(), trajectory_hash_seconds=callback.hash_seconds,
                parent_cpu_seconds=cpu_seconds, worker_memory=list(memories.values()),
                completed_episodes=callback.completed, outcomes=callback.counts)
        finally:
            if vec is not None:
                vec.close()
    result['resources'] = sampler.result
    io.write(folder / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('setting', 'steps_per_second', 'resources')}), flush=True)
    return result


def same_run(rows):
    keys = ('steps', 'initial_parameters_sha256', 'final_parameters_sha256', 'trajectory_sha256')
    return all(len({row[key] for row in rows}) == 1 for key in keys)


def run(source, out):
    plan = freeze(source, out)
    io.write(out / 'runtime.json', dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat()))
    original = io.read(source / 'unmeasured_a/result.json')
    qualification = trial(plan, plan['qualification'], out)
    if not same_run([original, qualification]):
        raise AssertionError('Grouped4-env learning trajectory or parameters differ from original profile')
    io.write(out / 'qualification.json', dict(real_learning_steps=plan['steps_per_trial'],
        trajectories_and_final_parameters_equal=True, scope='Same4 virtual environments; original4 processes versus grouped2.'))
    results = []
    for setting in plan['trials']:
        gc.collect()
        io.verify(plan)
        results.append(trial(plan, setting, out))
    ordinary = [r for r in results if not r['setting']['grouped']]
    grouped = [r for r in results if r['setting']['grouped']]
    if not same_run([original, *ordinary]) or not same_run(grouped):
        raise AssertionError('Run-to-run learning replay changed within a topology')
    normal_time = median(r['learning_seconds'] for r in ordinary)
    grouped_time = median(r['learning_seconds'] for r in grouped)
    headroom = min(r['resources']['minimum_commit_headroom_gib'] for r in grouped)
    physical = min(r['resources']['minimum_available_memory_gib'] for r in grouped)
    faster = all(g['learning_seconds'] < b['learning_seconds'] for g, b in zip(grouped, ordinary))
    adopt = faster and grouped_time <= normal_time * .90 and headroom >= 2.8 and physical >= 2.
    analysis = dict(replay_qualification_passed=True, repeats_equal_within_topology=True,
        ordinary4_median_seconds=normal_time, grouped16_median_seconds=grouped_time,
        throughput_ratio=normal_time / grouped_time, grouped16_steps_per_second=plan['steps_per_trial'] / grouped_time,
        minimum_grouped_commit_headroom_gib=headroom, minimum_grouped_physical_headroom_gib=physical,
        candidate_execution_recommended=adopt, policy_promoted=False, scope=plan['scope'])
    io.write(out / 'analysis.json', analysis)
    io.verify(plan)
    io.write(out / 'completion.json', dict(status='benchmark_complete', diagnostic_steps=5 * plan['steps_per_trial'],
        candidate_execution_recommended=adopt, completed_at=datetime.now(timezone.utc).isoformat(), scope=plan['scope']))
    print(json.dumps(analysis), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.source.resolve(), args.out.resolve())
    except BaseException:
        io.write(args.out / 'failure.json', dict(error=traceback.format_exc(), pid=os.getpid(), at=datetime.now(timezone.utc).isoformat()))
        raise
