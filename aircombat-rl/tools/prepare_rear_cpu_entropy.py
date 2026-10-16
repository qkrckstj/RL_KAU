"""Prepare a bounded, single-factor CPU exploration follow-up after review.

This leaves the running/frozen trainer unchanged. It prepares an ordinary
rear_async_cpu resume plan with an exactly restored parent and ent_coef=0.01.
Previously measured identical parent evaluation panels are explicitly reused.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

from experiments.rear_async.storage import identity, read, sha, verify, write
from tools.rear_async_cpu import ROOT, check_plan


def tensor_digest(value):
    import torch
    h = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            a = x.detach().cpu().contiguous()
            h.update(str((str(a.dtype), tuple(a.shape))).encode())
            h.update(a.numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(repr(k).encode())
                visit(x[k])
        elif isinstance(x, (tuple, list)):
            for a in x:
                visit(a)
        else:
            h.update(repr(x).encode())

    visit(value)
    return h.hexdigest()


def prepare(source, out):
    import psutil
    import torch
    from stable_baselines3 import PPO

    if out.exists():
        raise FileExistsError('Use a fresh output directory')
    for name in ('training_complete.json', 'evaluation_complete.json'):
        if not (source / name).exists():
            raise RuntimeError('Previous native CPU run is not terminal: ' + name)
    if read(source / 'evaluation_complete.json')['status'] != 'queue_drained':
        raise RuntimeError('Previous evaluation queue was not drained')
    for pid in read(source / 'processes.json').values():
        if psutil.pid_exists(pid):
            p = psutil.Process(pid)
            if p.status() != psutil.STATUS_ZOMBIE and 'tools.rear_async_cpu' in ' '.join(p.cmdline()):
                raise RuntimeError('Previous trainer/evaluator is still live')
    plan = read(source / 'plan.json')
    check_plan(plan)
    for seed in plan['seeds']:
        d = read(source / f's{seed}_budget_decision.json')
        if d['extend']:
            raise RuntimeError('Review extended continuation before changing exploration')
    # This hypothesis is specifically conditional on loss of parent rear skill.
    old_paths = list((source / 'evaluations').glob('*/completion.json'))
    old = [read(p) for p in old_paths]
    baseline = next(x for x in old if x['baseline'] and x['kind'] == 'development')
    endpoints = [x for x in old if not x['baseline'] and x['kind'] == 'development']
    if len(endpoints) != 2 or any(x['summary']['rear_damage'] >= baseline['summary']['rear_damage'] for x in endpoints):
        raise RuntimeError('Rear skill is not lower in both broad endpoints; reconsider hypothesis')
    if psutil.virtual_memory().available < 10 * 2**30:
        raise MemoryError('Available host memory below 10 GiB')
    bands = [1820400000, 1830400000]
    reservations = [ROOT / f'runs/training_reservation_{b}.json' for b in bands]
    if any(p.exists() for p in reservations):
        raise FileExistsError('Follow-up training band already reserved')
    torch.set_num_threads(1)
    parent = PPO.load(ROOT / plan['parent'], device='cpu')
    before = dict(policy=tensor_digest(parent.policy.state_dict()),
                  optimizer=tensor_digest(parent.policy.optimizer.state_dict()),
                  num_timesteps=parent.num_timesteps)
    baseline_checkpoint = Path(read(source / 'baseline_checkpoint.json')['path'])
    verify(baseline_checkpoint)
    cached_parent = PPO.load(baseline_checkpoint / 'learner.zip', device='cpu')
    if tensor_digest(cached_parent.policy.state_dict()) != before['policy']:
        raise ValueError('Cached parent evaluation is from a different policy')
    from experiments.rear_async.evaluate import profile
    for path in old_paths:
        entry = read(path)
        if not entry['baseline']:
            continue
        verify(path.parent / 'input')
        rows = [read(p) for p in (path.parent / 'cells').glob('*.json')]
        kind = entry['kind']
        band = plan['monitor_band'] if kind == 'monitor' else plan['development_band']
        n = plan['monitor_ics'] if kind == 'monitor' else plan['development_ics']
        foes = plan['monitor_opponents'] if kind == 'monitor' else plan['development_opponents']
        replicas = [4900] if kind == 'monitor' else [4900, 4901]
        expected = {(f['id'], band+i, seat, a) for f in foes for i in range(n)
                    for seat in ('red', 'blue') for a in replicas}
        actual = {(x['opponent'], x['seed'], x['seat'], x['action_seed']) for x in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError('Cached parent panel identity differs')
        if not all(math.isclose(profile(rows)[k], v, abs_tol=1e-12)
                   for k, v in entry['summary'].items()):
            raise ValueError('Cached parent summary does not match its raw cells')
    if parent.ent_coef != 0:
        raise ValueError('Expected the unmodified entropy-zero parent')
    out.mkdir(parents=True)
    parent.ent_coef = .01
    prepared = out / 'exploration_parent.zip'
    parent.save(prepared)
    restored = PPO.load(prepared, device='cpu')
    after = dict(policy=tensor_digest(restored.policy.state_dict()),
                 optimizer=tensor_digest(restored.policy.optimizer.state_dict()),
                 num_timesteps=restored.num_timesteps)
    if before != after or restored.ent_coef != .01:
        raise ValueError('Parent actor/critic/Adam/counter did not restore exactly')
    original_parent = plan['parent']
    plan = copy.deepcopy(plan)
    plan.update(parent=str(prepared.relative_to(ROOT)), seeds=[42200, 42201],
                training_bands=bands, parent_source=original_parent,
                intervention=dict(factor='ppo.ent_coef', control=0., value=.01),
                control_run=str(source),
                comparison_scope='Same parent, budget, native evaluation conditions and other settings. Fresh continuation RNGs; not paired training trajectories or confirmed causal superiority.')
    plan['ppo']['ent_coef'] = .01
    plan['frozen_sha256'][str(prepared.relative_to(ROOT))] = sha(prepared)
    plan['frozen_sha256'][str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    plan['config_id'] = identity({k: v for k, v in plan.items() if k not in ('frozen_sha256', 'config_id')})
    write(out / 'plan.json', plan)
    for p, band in zip(reservations, bands):
        write(p, dict(run=str(out), start=band, stop_exclusive=band + 1000000))
    # ent_coef affects training only. State/export/physics and evaluation
    # conditions are unchanged, so the two measured parent panels are reusable.
    for entry in [x for x in old if x['baseline']]:
        key = identity(dict(reused_baseline=entry['kind'], source=str(source), config_id=plan['config_id']))
        reused = dict(entry, actual_new_games=0, new_games_in_final_attempt=0,
                      reused_cells_in_final_attempt=entry['summary']['games'],
                      wall_seconds=0., reused_from_run=str(source),
                      original_evaluation_seconds=entry['wall_seconds'],
                      reuse_basis='Exact parent policy/optimizer state; training-only entropy change; same frozen native evaluation source and fixed panel.')
        write(out / 'evaluations' / key / 'completion.json', reused)
        write(out / 'candidates' / f'{key}.json', reused)
    write(out / 'baseline_checkpoint.json', read(source / 'baseline_checkpoint.json'))
    write(out / 'parent_verification.json', dict(before=before, after=after, exact=True,
          original_parent_sha256=sha(ROOT / original_parent), prepared_parent_sha256=sha(prepared),
          intervention=plan['intervention'], new_evaluation_games=0,
          reused_baseline_games=sum(x['summary']['games'] for x in old if x['baseline'])))
    write(out / 'started.json', dict(time=time.time(), status='prepared_not_launched'))
    check_plan(plan)
    print(json.dumps(dict(out=str(out), status='prepared', exact_parent_restore=True,
                          intervention=plan['intervention'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    prepare(a.source.resolve(), a.out.resolve())
