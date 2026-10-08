"""Fresh-process performance audit of the actual RL_KAU learners and CEM games.

No archived sources, checkpoints, replay buffers, or final test bands are changed.
PPO with two environments is an explicit throughput experiment, not an equivalent
learning trajectory. Global rollout size and minibatch size are preserved.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from tools.plan_a import ROOT, timestamp, write_json

WARM_PPO = ROOT / 'runs/autolab_20261005/ppo_pilot/s700/checkpoints/step_0/policy_net.zip'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_hash(model):
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                  for p in model.policy.parameters())).hexdigest()


def neural_case(algorithm, device, steps, seed, n_envs=1, cpu_rollout=False):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from experiments.plan_a.core import make_env, make_learner, specification
    from tools.plan_a import set_exploration_duration
    from tools.ppo_cpu_rollouts import cpu_rollouts

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    if n_envs not in (1, 2) or (n_envs != 1 and not algorithm.startswith('ppo')):
        raise ValueError('Only the explicit PPO experiment supports two environments')
    if cpu_rollout and not algorithm.startswith('ppo'):
        raise ValueError('CPU rollout mode requires PPO')
    if steps < 1024 or steps % 1024:
        raise ValueError('Use at least 1024 steps, divisible by 1024')
    if not algorithm.startswith('ppo') and steps < 20480:
        raise ValueError('DQN audits require >=20480 steps to preserve the exploration schedule')
    torch.set_num_threads(1)
    spec = specification('observed_ppo' if algorithm.startswith('ppo') else 'observed_dqn')
    if algorithm == 'ddqn':
        spec['double_dqn'] = True
    checkpoint_hash = sha(WARM_PPO) if algorithm == 'ppo_warm' else None

    def build():
        if algorithm == 'ppo_warm':
            # This step-zero archive already contains the actual transferred actor
            # and fresh critic; loading it does not require the missing replay.
            config = json.loads((WARM_PPO.parents[2] / 'config.json').read_text())
            if sha(WARM_PPO) != checkpoint_hash:
                raise ValueError('Initial PPO checkpoint changed')
            env = Monitor(make_env(config['spec'], seed=seed)) if n_envs == 1 else SubprocVecEnv(
                [partial(make_env, config['spec'], seed=seed+i) for i in range(n_envs)], start_method='spawn')
            try:
                model = PPO.load(WARM_PPO, env=env, device=device, seed=seed,
                                 n_steps=2048 // n_envs)
            except BaseException:
                env.close()
                raise
        else:
            model = make_learner(spec, seed, device)
            if n_envs != 1:
                model.get_env().close()
                env = SubprocVecEnv([partial(make_env, spec, seed=seed+i) for i in range(n_envs)],
                                   start_method='spawn')
                model = PPO('MlpPolicy', env, seed=seed, device=device, gamma=spec['gamma'],
                            policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.ReLU),
                            n_steps=1024 // n_envs, batch_size=64, n_epochs=10,
                            gae_lambda=.95, ent_coef=.01, target_kl=.03)
        return model

    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()

    warm = build()
    try:
        with cpu_rollouts(warm) if cpu_rollout else nullcontext():
            warm.learn(2048 if algorithm == 'ppo_warm' else 1024)
        sync()
    finally:
        warm.get_env().close()
    del warm
    model = build()
    env = model.get_env()
    if not algorithm.startswith('ppo'):
        set_exploration_duration(model, steps, 20480)
    initial = weights_hash(model)
    totals = dict(rollout=0.0, environment=0.0, update=0.0)
    optimizer_steps = 0
    originals = (env.step_wait, model.collect_rollouts, model.train, model.policy.optimizer.step)

    def timed(name, fn):
        def call(*a, **kw):
            start = time.perf_counter()
            result = fn(*a, **kw)
            if name != 'environment':
                sync()
            totals[name] += time.perf_counter() - start
            return result
        return call

    def optimizer_step(*a, **kw):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return originals[3](*a, **kw)

    try:
        env.step_wait = timed('environment', originals[0])
        model.collect_rollouts = timed('rollout', originals[1])
        model.train = timed('update', originals[2])
        model.policy.optimizer.step = optimizer_step
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        sync()
        start = time.perf_counter()
        with cpu_rollouts(model) if cpu_rollout else nullcontext():
            model.learn(steps)
        sync()
        elapsed = time.perf_counter() - start
        final = weights_hash(model)
        correct_device = all(p.device.type == device for p in model.policy.parameters())
        finite = all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())
        if not correct_device or not finite or optimizer_steps == 0 or initial == final:
            raise RuntimeError('Actual finite weight updates on the requested device were not verified')
        return dict(algorithm=algorithm, device=device, n_envs=n_envs, seed=seed,
                    cpu_rollouts=cpu_rollout,
                    requested_steps=steps, actual_steps=model.num_timesteps,
                    optimizer_steps=optimizer_steps, train_seconds=elapsed,
                    steps_per_second=model.num_timesteps/elapsed,
                    rollout_seconds=totals['rollout'], environment_seconds=totals['environment'],
                    update_seconds=totals['update'], initial_weights_sha256=initial,
                    final_weights_sha256=final, all_parameters_on_requested_device=correct_device,
                    torch=torch.__version__, cuda_build=torch.version.cuda,
                    gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                    checkpoint=str(WARM_PPO.relative_to(ROOT)) if algorithm == 'ppo_warm' else None,
                    checkpoint_sha256=sha(WARM_PPO) if algorithm == 'ppo_warm' else None,
                    config=dict(batch_size=model.batch_size, gamma=model.gamma,
                                rollout_steps=getattr(model, 'n_steps', None),
                                epochs=getattr(model, 'n_epochs', None),
                                target_kl=getattr(model, 'target_kl', None)))
    finally:
        env.step_wait, model.collect_rollouts, model.train, model.policy.optimizer.step = originals
        env.close()


def cem_case(out, workers, jobs):
    from tools.league_matches import evaluate_jobs
    start = time.perf_counter()
    results = evaluate_jobs(jobs, out, workers=workers)
    elapsed = time.perf_counter() - start
    canonical = [{k: v for k, v in r.items() if k != 'elapsed_seconds'} for r in results]
    return dict(workers=workers, seconds=elapsed, games=sum(r['summary']['n'] for r in results),
                steps=sum(e['steps'] for r in results for e in r['episodes']),
                outcomes_sha256=hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--algorithms', nargs='+', choices=('dqn', 'ddqn', 'ppo', 'ppo_warm'),
                   default=['dqn', 'ddqn', 'ppo', 'ppo_warm'])
    p.add_argument('--devices', nargs='+', choices=('cpu', 'cuda'), default=['cpu', 'cuda'])
    p.add_argument('--steps', type=int, default=24576)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--n-envs', type=int, choices=(1, 2), default=1)
    p.add_argument('--cpu-rollouts', action='store_true', help='PPO-only CPU collection, requested device for updates')
    p.add_argument('--cem', action='store_true', help='also compare 1/3 CPU workers on identical paired games')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--seed', type=int, default=0, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.steps < 2048 or a.steps % 2048 or a.repeats < 1:
        p.error('steps must be >=2048 and divisible by 2048; repeats must be positive')
    if a.n_envs != 1 and any(not kind.startswith('ppo') for kind in a.algorithms):
        p.error('two environments are an explicit PPO-only experiment')
    if a.cpu_rollouts and any(not kind.startswith('ppo') for kind in a.algorithms):
        p.error('CPU rollout mode is PPO-only')
    if a.worker:
        print('RESULT ' + json.dumps(neural_case(a.algorithms[0], a.devices[0], a.steps, a.seed, a.n_envs, a.cpu_rollouts)))
        return 0
    a.out.mkdir(parents=True, exist_ok=False)
    source_names = ['experiments/plan_a/core.py', 'tools/plan_a.py', 'tools/autolab_ppo.py',
                    'tools/runtime_audit.py', 'tools/ppo_cpu_rollouts.py', 'tools/league_matches.py',
                    'tools/grade.py', 'experiments/league/controller.py',
                    'experiments/league/bundle/models/final/policy_net.json']
    source_names += [str(path.relative_to(ROOT)) for path in sorted((ROOT/'aircombat_gym').rglob('*.py'))]
    if 'ppo_warm' in a.algorithms:
        source_names += [str(WARM_PPO.relative_to(ROOT)),
                         str((WARM_PPO.parents[2]/'config.json').relative_to(ROOT))]
    sources = {name: sha(ROOT/name) for name in source_names}
    report = dict(started_utc=timestamp(), platform=platform.platform(), source_sha256=sources,
                  scope='Throughput only; no model selection or final test; warmup discarded; fresh process per case',
                  n_envs=a.n_envs, cpu_rollouts=a.cpu_rollouts,
                  parallel_ppo_changes_sampling=a.n_envs != 1,
                  environment_timing='step_wait wall time, including reset; SubprocVecEnv includes IPC/wait, not total simulator CPU time',
                  runs=[])
    for name in sources:
        import shutil
        target = a.out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    for seed in range(a.repeats):
        for kind in a.algorithms:
            for device in a.devices[::1 if seed % 2 == 0 else -1]:
                label = f'{kind}_{device}_env{a.n_envs}_s{seed}'
                command = [sys.executable, '-m', 'tools.runtime_audit', '--worker', '--out', str(a.out),
                           '--algorithms', kind, '--devices', device, '--steps', str(a.steps),
                           '--seed', str(seed), '--n-envs', str(a.n_envs)]
                if a.cpu_rollouts:
                    command.append('--cpu-rollouts')
                print('START ' + label, flush=True)
                proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                      env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                (a.out/f'{label}.log').write_text(proc.stdout+'\n'+proc.stderr)
                if proc.returncode:
                    raise RuntimeError(f'Worker failed: {a.out/label}.log')
                result = json.loads(next(line[7:] for line in proc.stdout.splitlines() if line.startswith('RESULT ')))
                report['runs'].append(result)
                print(f"DONE {label}: {result['train_seconds']:.3f}s, {result['optimizer_steps']} updates", flush=True)
                write_json(a.out/'results.json', report)
    for kind in a.algorithms:
        for seed in range(a.repeats):
            group = [r for r in report['runs'] if r['algorithm'] == kind and r['seed'] == seed]
            if len({r['initial_weights_sha256'] for r in group}) != 1:
                raise RuntimeError('Initial weights differed across devices')
            if len({(r['actual_steps'], r['optimizer_steps']) for r in group}) != 1:
                raise RuntimeError('Step/update budgets differed across devices; do not claim an equal-work speedup')
    report['summary'] = {f'{kind}_{device}': statistics.median(
        r['train_seconds'] for r in report['runs'] if r['algorithm'] == kind and r['device'] == device)
        for kind in a.algorithms for device in a.devices}
    if a.cem:
        from tools.autolab_cem import read
        own = dict(id='frozen_final', kind='reactive', parameters=read(
            ROOT/'experiments/league/bundle/models/final/policy_net.json')['parameters'])
        # Previously consumed development conditions, never described as unseen test data.
        jobs = [dict(own=own, foe=dict(id=name, kind='bot', name=name), band=33030000+i, n=2)
                for i in range(4) for name in ('ace', 'pursuit', 'evader')]
        report['cem'] = []
        write_json(a.out/'cem_jobs.json', jobs)
        for repeat in range(a.repeats):
            for workers in (1, 3)[::1 if repeat % 2 == 0 else -1]:
                result = cem_case(a.out/f'cem_w{workers}_r{repeat}', workers, jobs)
                report['cem'].append(dict(repeat=repeat, **result))
                print(f"CEM {workers} workers: {result['seconds']:.3f}s", flush=True)
        if len({r['outcomes_sha256'] for r in report['cem']}) != 1:
            raise RuntimeError('CEM game outcomes changed across worker counts')
    if any(sha(ROOT/name) != digest for name, digest in sources.items()):
        raise RuntimeError('Source changed during audit')
    report.update(finished_utc=timestamp(), equal_initial_weights_and_budgets=True)
    write_json(a.out/'results.json', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
