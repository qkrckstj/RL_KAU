"""Prepare a fresh bounded entropy continuation from a measured rear candidate.

The frozen trainer, prior outputs and selected learner remain unchanged.
Only training entropy changes; exact selected-parent panels are reused.
"""
import argparse
import copy
import fcntl
import json
import math
import time
from pathlib import Path

from experiments.rear_async.evaluate import profile
from experiments.rear_async.storage import identity, read, sha, verify, write
from tools.prepare_rear_cpu_entropy import tensor_digest
from tools.rear_async_cpu import ROOT, check_plan


def validate_panel(plan, checkpoint, folder):
    selected = verify(checkpoint)
    cached = verify(folder / 'input')
    if cached != selected:
        raise ValueError('Cached learner/export differs from selected checkpoint')
    entry = read(folder / 'completion.json')
    if entry['checkpoint_id'] != selected['checkpoint_id']:
        raise ValueError('Cached checkpoint identity differs')
    kind = entry['kind']
    if kind not in ('monitor', 'development'):
        raise ValueError('Unknown evaluation panel')
    band, n = plan[kind + '_band'], plan[kind + '_ics']
    foes = plan[kind + '_opponents']
    replicas = [4900] if kind == 'monitor' else [4900, 4901]
    expected = {(f['id'], band + i, seat, a) for f in foes for i in range(n)
                for seat in ('red', 'blue') for a in replicas}
    rows = [read(p) for p in (folder / 'cells').glob('*.json')]
    actual = {(x['opponent'], x['seed'], x['seat'], x['action_seed']) for x in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError('Cached panel cells differ')
    summaries = [(profile(rows), entry['summary'])]
    summaries += [(profile([r for r in rows if r['opponent'] == f['id']]),
                   entry['per_opponent'][f['id']]) for f in foes]
    for measured, reported in summaries:
        if measured.keys() != reported.keys() or not all(
                math.isclose(measured[k], v, rel_tol=0, abs_tol=1e-12)
                for k, v in reported.items()):
            raise ValueError('Cached summary does not match raw cells')
    return entry


def reused_baseline(entry, folder, source, seed):
    return dict(entry, seed=seed, baseline=True, additional_steps=0,
                source_seed=entry['seed'], source_additional_steps=entry['additional_steps'],
                win_gain=None, actual_new_games=0, new_games_in_final_attempt=0,
                reused_cells_in_final_attempt=entry['summary']['games'], wall_seconds=0.,
                original_evaluation_seconds=entry['wall_seconds'],
                reused_from_run=str(source), reused_from_evaluation=str(folder),
                candidate=True, status='reused_selected_parent_baseline',
                reuse_basis='Exact checkpoint/exports; unchanged frozen native evaluator and panel; entropy affects training only.')


def prepare(source, checkpoint, out, entropy, seeds, bands):
    import psutil
    import torch
    from stable_baselines3 import PPO

    if out.exists():
        raise FileExistsError('Use a fresh output directory')
    if not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Output must be in this worktree runs directory')
    if not (source / 'training_complete.json').exists() or (source / 'training_failure.json').exists():
        raise RuntimeError('Previous training is not successfully terminal')
    if read(source / 'evaluation_complete.json')['status'] != 'queue_drained':
        raise RuntimeError('Previous evaluations are not complete')
    if len(seeds) != 2 or len(set(seeds)) != 2 or len(bands) != 2 or len(set(bands)) != 2:
        raise ValueError('Use two distinct continuation seeds and training bands')
    if not math.isfinite(entropy) or entropy < 0:
        raise ValueError('Entropy must be finite and nonnegative')
    if psutil.virtual_memory().available < 10 * 2**30:
        raise MemoryError('Available host memory below10GiB')
    for pid in read(source / 'processes.json').values():
        if psutil.pid_exists(pid):
            p = psutil.Process(pid)
            if p.status() != psutil.STATUS_ZOMBIE and 'tools.rear_async_cpu' in ' '.join(p.cmdline()):
                raise RuntimeError('Previous native trainer/evaluator is live')
    locks = []
    try:
        for name in ('training.lock', 'evaluation.lock'):
            lock = (source / name).open('a')
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(source / 'plan.json')
        check_plan(plan)
        if set(seeds) & set(plan['seeds']):
            raise ValueError('Continuation RNGs must be fresh')
        meta = verify(checkpoint)
        if checkpoint.parent != source / 'checkpoints' or meta['config_id'] != plan['config_id']:
            raise ValueError('Selected checkpoint is outside source experiment')
        if meta['baseline'] or meta['additional_steps'] < plan['minimum_steps']:
            raise ValueError('Select a measured candidate after the minimum budget')
        evaluations = []
        for kind in ('monitor', 'development'):
            matches = [p.parent for p in (source / 'evaluations').glob('*/completion.json')
                       if read(p)['checkpoint_id'] == meta['checkpoint_id'] and read(p)['kind'] == kind]
            if len(matches) != 1:
                raise ValueError('Need exactly one completed selected-parent panel per kind')
            folder = matches[0]
            evaluations.append((folder, validate_panel(plan, checkpoint, folder)))
        # Inspect reservations across sibling worktrees as well as this run.
        repository = ROOT.parent.parent.parent
        for file in repository.glob('**/training_reservation_*.json'):
            reservation = read(file)
            if any(b < reservation['stop_exclusive'] and b + 1000000 > reservation['start'] for b in bands):
                raise FileExistsError('Training band overlaps reservation: ' + str(file))
        if abs(bands[0] - bands[1]) < 1000000:
            raise ValueError('New training bands overlap')
        torch.set_num_threads(1)
        parent = PPO.load(checkpoint / 'learner.zip', device='cpu')
        before = dict(policy=tensor_digest(parent.policy.state_dict()),
                      optimizer=tensor_digest(parent.policy.optimizer.state_dict()),
                      num_timesteps=parent.num_timesteps)
        old_entropy = parent.ent_coef
        if old_entropy != plan['ppo']['ent_coef'] or old_entropy == entropy:
            raise ValueError('Entropy intervention does not match source settings')
        out.mkdir(parents=True)
        parent.ent_coef = entropy
        prepared = out / 'selected_parent.zip'
        parent.save(prepared)
        restored = PPO.load(prepared, device='cpu')
        after = dict(policy=tensor_digest(restored.policy.state_dict()),
                     optimizer=tensor_digest(restored.policy.optimizer.state_dict()),
                     num_timesteps=restored.num_timesteps)
        if before != after or restored.ent_coef != entropy:
            raise ValueError('Selected actor/critic/Adam/counter did not restore exactly')
        plan = copy.deepcopy(plan)
        plan.update(parent=str(prepared.relative_to(ROOT)), seeds=seeds, training_bands=bands,
                    parent_source=str((checkpoint / 'learner.zip').relative_to(ROOT)),
                    selected_parent_checkpoint=str(checkpoint), source_run=str(source),
                    intervention=dict(factor='ppo.ent_coef', control=old_entropy, value=entropy),
                    comparison_scope='Selected rear specialist continuation with fresh training RNGs. Not paired training trajectories. Compare selected parent AND preserved geometry/best-win candidates on consumed development conditions.')
        plan['ppo']['ent_coef'] = entropy
        for path in (prepared, Path(__file__).resolve(), checkpoint / 'learner.zip'):
            plan['frozen_sha256'][str(path.relative_to(ROOT))] = sha(path)
        plan['config_id'] = identity({k: v for k, v in plan.items() if k not in ('config_id', 'frozen_sha256')})
        write(out / 'plan.json', plan)
        for band in bands:
            reservation_path = ROOT / f'runs/training_reservation_{band}.json'
            with reservation_path.open('x') as f:
                json.dump(dict(run=str(out), start=band, stop_exclusive=band + 1000000), f)
        for folder, entry in evaluations:
            key = identity(dict(reused_baseline=entry['kind'], source=str(folder), config_id=plan['config_id']))
            reused = reused_baseline(entry, folder, source, seeds[0])
            write(out / 'evaluations' / key / 'completion.json', reused)
            write(out / 'candidates' / f'{key}.json', reused)
        write(out / 'baseline_checkpoint.json', dict(path=str(checkpoint)))
        write(out / 'parent_verification.json', dict(before=before, after=after, exact=True,
              selected_parent_sha256=sha(checkpoint / 'learner.zip'), prepared_parent_sha256=sha(prepared),
              intervention=plan['intervention'], new_evaluation_games=0,
              reused_baseline_games=sum(e['summary']['games'] for _, e in evaluations),
              selected_parent_development=next(e['summary'] for _, e in evaluations if e['kind'] == 'development')))
        write(out / 'started.json', dict(time=time.time(), status='prepared_not_launched'))
        check_plan(plan)
        print(json.dumps(dict(out=str(out), status='prepared', exact_parent_restore=True,
                              intervention=plan['intervention'], reused_games=896)))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('source', 'checkpoint', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--entropy', type=float, required=True)
    p.add_argument('--seeds', type=int, nargs=2, required=True)
    p.add_argument('--bands', type=int, nargs=2, required=True)
    a = p.parse_args()
    prepare(a.source.resolve(), a.checkpoint.resolve(), a.out.resolve(), a.entropy, a.seeds, a.bands)
