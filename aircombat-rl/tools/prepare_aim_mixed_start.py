"""Prepare the bounded reset-mixture comparison without replaying its baseline."""
import argparse
import copy
import fcntl
import json
import math
import shutil
import time
from pathlib import Path

from experiments.aim_async.evaluate import profile
from experiments.rear_async.storage import identity, read, sha, verify, write
from tools.prepare_rear_cpu_entropy import tensor_digest
from tools.prepare_rear_cpu_selected_parent import reused_baseline
from tools.rear_async_cpu import ROOT, check_plan


def validate_cached_panel(plan, folder):
    entry = read(folder / 'completion.json')
    kind = entry['kind']
    if not entry['baseline'] or kind not in ('monitor', 'development'):
        raise ValueError('Expected a completed baseline panel')
    replicas = (4900,) if kind == 'monitor' else (4900, 4901)
    expected = {(f['id'], plan[kind+'_band']+i, seat, a)
                for f in plan[kind+'_opponents'] for i in range(plan[kind+'_ics'])
                for seat in ('red', 'blue') for a in replicas}
    files = sorted((folder / 'cells').glob('*.json'))
    rows = [read(f) for f in files]
    keys = {(r['opponent'], r['seed'], r['seat'], r['action_seed']) for r in rows}
    if keys != expected or len(rows) != len(expected):
        raise ValueError('Baseline raw cell identities differ')
    summaries = [(profile(rows), entry['summary'])]
    summaries += [(profile([r for r in rows if r['opponent'] == foe['id']]),
                   entry['per_opponent'][foe['id']]) for foe in plan[kind+'_opponents']]
    for measured, reported in summaries:
        if measured.keys() != reported.keys() or any(
                measured[k] != v and not (measured[k] is not None and v is not None and
                math.isclose(measured[k], v, rel_tol=0., abs_tol=1e-12))
                for k, v in reported.items()):
            raise ValueError('Baseline summary differs from raw cells')
    return entry, rows, files


def prepare(source, out):
    import psutil
    import torch
    from stable_baselines3 import PPO
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Use a fresh output under runs')
    if not (source / 'training_complete.json').exists() or (source / 'training_failure.json').exists():
        raise RuntimeError('Source training is not successfully terminal')
    if read(source / 'evaluation_complete.json')['status'] != 'queue_drained':
        raise RuntimeError('Source evaluation queue is not drained')
    for process in psutil.process_iter(['cmdline', 'status']):
        cmd = process.info['cmdline'] or []
        if process.info['status'] != psutil.STATUS_ZOMBIE and 'launch' in cmd and any(
                Path(arg).name == 'aim_cpu_runner.py' or arg == 'tools.rear_async_cpu' for arg in cmd):
            raise RuntimeError('Existing native trainer: ' + str(process.pid))
    if psutil.virtual_memory().available < 10 * 2**30:
        raise MemoryError('Need10GiB available')
    locks = []
    try:
        for name in ('launch.lock', 'training.lock', 'evaluation.lock'):
            lock = (source/name).open('a'); locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = read(source/'plan.json'); check_plan(original)
        if original['curriculum'] != dict(stage='disabled', fraction=0.):
            raise ValueError('This comparison requires the neutral-only aiming control')
        if any(read(source/f's{s}_budget_decision.json')['extend'] for s in original['seeds']):
            raise ValueError('Unexpected extended control; review before follow-up')
        evidence = read(source/'parent_verification.json')
        agreement = read(source/'baseline_native_agreement.json')
        parent = ROOT/original['parent']
        if not evidence['exact_original_parameters'] or sha(parent) != evidence['prepared_sha256']:
            raise ValueError('Prepared parent changed')
        if not agreement['exact'] or agreement['games'] != 128:
            raise ValueError('Missing exact migrated native baseline agreement')
        checkpoint = Path(read(source/'baseline_checkpoint.json')['path'])
        meta = verify(checkpoint)
        if not meta['baseline'] or meta['config_id'] != original['config_id']:
            raise ValueError('Source baseline checkpoint differs')
        panels = []
        for kind in ('monitor', 'development'):
            folders = [p.parent for p in (source/'evaluations').glob('*/completion.json')
                       if read(p)['baseline'] and read(p)['kind'] == kind]
            if len(folders) != 1:
                raise ValueError('Ambiguous source baseline')
            entry, rows, files = validate_cached_panel(original, folders[0])
            if entry['checkpoint_id'] != meta['checkpoint_id']:
                raise ValueError('Baseline model identity changed')
            if kind == 'monitor' and verify(folders[0]/'input') != meta:
                raise ValueError('Baseline evaluated input differs')
            panels.append((folders[0], entry, rows, files))
        seeds = [45400, 45401]; bands = [2460400000, 2470400000]
        for file in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            reservation = read(file)
            if any(b < reservation['stop_exclusive'] and b+1000000 > reservation['start'] for b in bands):
                raise ValueError('Training band reserved: ' + str(file))
        torch.set_num_threads(1)
        model = PPO.load(parent, device='cpu')
        baseline = PPO.load(checkpoint/'learner.zip', device='cpu')
        def state(m):
            return dict(policy=tensor_digest(m.policy.state_dict()), optimizer=tensor_digest(m.policy.optimizer.state_dict()),
                        decisions=m.num_timesteps, updates=m._n_updates)
        if state(model) != state(baseline) or model.observation_space.shape != (43,):
            raise ValueError('Prepared parent differs from evaluated baseline')
        out.mkdir(parents=True)
        prepared = out/'selected_parent.zip'; shutil.copyfile(parent, prepared)
        if sha(prepared) != sha(parent):
            raise ValueError('Parent copy changed')
        runner_text = (source/'aim_cpu_runner.py').read_text()
        old = 'from experiments.aim_async.worker import make_worker'
        if runner_text.count(old) != 1:
            raise ValueError('Worker integration boundary changed')
        runner = out/'aim_cpu_runner.py'
        runner.write_text(runner_text.replace(old, 'from experiments.aim_async.mixed_start_worker import make_worker'))
        plan = copy.deepcopy(original)
        plan.update(parent=str(prepared.relative_to(ROOT)), parent_source=str(parent.relative_to(ROOT)),
                    source_run=str(source), seeds=seeds, training_bands=bands,
                    curriculum=dict(stage='quarter', fraction=.5),
                    intervention=dict(factor='training reset mixture', control='100% native neutral', value='50% native neutral +50% preserved quarter starts',
                                      unchanged='learner, Adam, aim features/reward, PPO, opponents, evaluation panels'),
                    comparison_scope='Adaptive two-seed distribution comparison. Fresh RNGs, not paired training trajectories; consumed development, not unseen final validation.')
        inputs = [prepared, runner, Path(__file__).resolve(), ROOT/'experiments/aim_async/mixed_start_worker.py',
                  source/'plan.json', source/'parent_verification.json', source/'baseline_native_agreement.json', checkpoint/'READY.json']
        for folder, entry, rows, files in panels:
            inputs += [folder/'completion.json', *files]
        plan['frozen_sha256'].update({str(p.relative_to(ROOT)): sha(p) for p in inputs})
        plan['config_id'] = identity({k:v for k,v in plan.items() if k not in ('config_id', 'frozen_sha256')})
        write(out/'plan.json', plan)
        for folder, entry, rows, files in panels:
            key = identity(dict(source=str(folder), config_id=plan['config_id']))
            reuse = reused_baseline(entry, folder, source, seeds[0])
            reuse['reuse_basis'] = 'Byte-identical43-feature learner; exact actor/critic/Adam/counters match evaluated baseline. Native evaluator and paired roster/ICs/seats/action seeds unchanged. Only training starts change.'
            write(out/'evaluations'/key/'completion.json', reuse)
            write(out/'candidates'/f'{key}.json', reuse)
            for row in rows:
                cell_id = identity({k:row[k] for k in ('opponent','seed','seat','action_seed')})
                write(out/'evaluations'/key/'cells'/f'{cell_id}.json', row)
        write(out/'baseline_checkpoint.json', dict(path=str(checkpoint)))
        write(out/'parent_verification.json', dict(exact=True, state=state(model), source_sha256=sha(parent),
              prepared_sha256=sha(prepared), reused_baseline_games=sum(len(p[2]) for p in panels), new_baseline_games=0))
        for band in bands:
            with (ROOT/f'runs/training_reservation_{band}.json').open('x') as f:
                json.dump(dict(run=str(out),start=band,stop_exclusive=band+1000000),f)
        write(out/'started.json', dict(time=time.time(),status='prepared_not_launched'))
        write(out/'goal_control.json', dict(status='prepared', automatic_learning=True, publication=False,
              effective_objective=plan['objective'], next_action='One read-only review, then launch the two fresh repetitions.'))
        check_plan(original); check_plan(plan)
        print(json.dumps(dict(out=str(out),prepared=True,reused_baseline_games=800,new_baseline_games=0,intervention=plan['intervention'])))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.source.resolve(), args.out.resolve())
