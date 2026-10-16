"""Prepare a fresh adaptive iteration without changing any frozen old source."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import time
from experiments.rear_async.storage import identity, read, sha, verify, write
from tools.aim_known_qualification import index_panel
from tools.aim_known_continuation import assert_profile, completed_panel
from tools.aim_known_mutual_focus import continuation_iteration, scores_from_rows


def freeze_sources(files, root):
    return {os.path.relpath(p.resolve(), root): sha(p) for p in files}


def prepare(source, qualification, role, out, seed, band):
    import psutil
    import torch
    from experiments.aim_async.evaluate import profile
    from experiments.aim_async.reference_ppo import ReferenceAnchoredPPO
    from tools.aim_prior_bias import learner_state
    from tools.prepare_aim_known_long import terminal
    from tools.rear_async_cpu import ROOT, check_plan
    if out.exists() or not out.is_relative_to(ROOT / 'runs') or seed < 0 or band < 0:
        raise ValueError('Require a fresh runs namespace and nonnegative seeds')
    if psutil.virtual_memory().available < 10 * 2**30:
        raise MemoryError('Need10GiB available')
    for proc in psutil.process_iter(['cmdline', 'status']):
        args = proc.info['cmdline'] or []
        if proc.info['status'] != psutil.STATUS_ZOMBIE and any(Path(a).name == 'aim_cpu_runner.py' for a in args):
            raise RuntimeError('Training/evaluation runner already active')
    locks = []
    try:
        for file in (source / 'launch.lock', source / 'training.lock', source / 'evaluation.lock',
                     qualification / 'qualification.lock'):
            lock = file.open('a'); locks.append(lock); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        terminal(source)
        original = read(source / 'plan.json'); check_plan(original)
        allocation_reference_path = Path(original['source_run']) / 'plan.json'
        reference_key = os.path.relpath(allocation_reference_path, ROOT)
        if original['frozen_sha256'].get(reference_key) != sha(allocation_reference_path):
            raise ValueError('Original pre-focus allocation reference is not authenticated')
        allocation_reference = read(allocation_reference_path)
        if len(original['opponents']) != 54 or original['minimum_steps'] != original['maximum_steps']:
            raise ValueError('Expected completed fixed-budget54-opponent source')
        manifest = read(qualification / 'manifest.json')
        if Path(manifest['run']).resolve() != source:
            raise ValueError('Qualification belongs to a different training run')
        checkpoint = Path(manifest[role]).resolve(); metadata = verify(checkpoint)
        rows54, evidence_files = completed_panel(qualification, checkpoint, original)
        audit = read(source / 'CYCLE_LOG_AUDIT.json')
        if set(audit['training']) != {str(s) for s in original['seeds']} or any(
                v['nonfinite_ppo_fields'] for v in audit['training'].values()):
            raise ValueError('Completed source audit missing or nonfinite')
        panels = []
        for kind in ('monitor', 'development'):
            matches = [(p, read(p)) for p in (source / 'evaluations').glob('*/completion.json')
                       if read(p)['kind'] == kind and Path(read(p)['checkpoint']).resolve() == checkpoint]
            if len(matches) != 1:
                raise ValueError('Require one completed source panel: ' + kind)
            file, entry = matches[0]
            paths = sorted(file.parent.glob('cells/*.json')); rows = [read(p) for p in paths]
            samplers = original['monitor_action_seeds'] if kind == 'monitor' else [4900, 4901]
            index_panel(rows, [o['id'] for o in original[kind + '_opponents']],
                        original[kind + '_band'], original[kind + '_ics'], samplers)
            if entry['checkpoint_id'] != metadata['checkpoint_id']:
                raise ValueError('Source panel identity differs')
            assert_profile(profile(rows), entry['summary'])
            panels.append((file, entry, paths, rows))
        for file in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            reservation = read(file)
            if band < reservation['stop_exclusive'] and band + 1000000 > reservation['start']:
                raise ValueError('Training band already reserved')
        torch.set_num_threads(1)
        model = ReferenceAnchoredPPO.load(checkpoint / 'learner.zip', device='cpu')
        before = learner_state(model)
        if model.reference_policy_hash != original['reference_policy_sha256']:
            raise ValueError('Reference policy changed')
        out.mkdir(parents=True)
        prepared = out / 'unchanged_parent.zip'; shutil.copyfile(checkpoint / 'learner.zip', prepared)
        restored = ReferenceAnchoredPPO.load(prepared, device='cpu')
        if sha(prepared) != sha(checkpoint / 'learner.zip') or learner_state(restored) != before:
            raise ValueError('Initial archive or NN/Adam/counters changed')
        code = (source / 'aim_cpu_runner.py').read_text()
        old = 'from tools.aim_known_focus_reporting import update'
        new = 'from tools.aim_known_mutual_focus_reporting import update'
        if code.count(old) == 1: code = code.replace(old, new)
        elif code.count(new) != 1: raise ValueError('Unknown runner reporter boundary')
        runner = out / 'aim_cpu_runner.py'; compile(code, str(runner), 'exec'); runner.write_text(code)
        monitor_expected = out / 'baseline_monitor_expected.json'
        write(monitor_expected, panels[0][3])
        plan = continuation_iteration(original, scores_from_rows(rows54, original['opponents']),
            allocation_reference, parent=prepared.relative_to(ROOT), source=source, seed=seed,
            band=band, metadata=metadata, wins21=panels[1][1]['summary']['wins'],
            wins54=profile(rows54)['wins'], checkpoint=checkpoint,
            assessment=qualification / 'completion.json', monitor_expected=monitor_expected)
        plan.update(parent_source=str((checkpoint / 'learner.zip').relative_to(ROOT)),
                    baseline_all54_qualification=str(qualification), baseline_all54_checkpoint=str(checkpoint),
                    budget_spec_repository_relative='docs/KNOWN_MUTUAL_FOCUS_ITERATION_20261015.md')
        files = [Path(__file__), ROOT / 'tools/aim_known_mutual_focus.py',
                 ROOT / 'tools/aim_known_mutual_focus_reporting.py', ROOT / 'tools/aim_known_continuation.py',
                 ROOT.parent / 'docs/KNOWN_MUTUAL_FOCUS_ITERATION_20261015.md',
                 source / 'plan.json', allocation_reference_path, source / 'CYCLE_LOG_AUDIT.json', checkpoint / 'READY.json',
                 prepared, runner, monitor_expected, *evidence_files]
        for file, _, paths, _ in panels: files.extend([file, *paths])
        plan['frozen_sha256'].update(freeze_sources(files, ROOT))
        plan['config_id'] = identity({k: v for k, v in plan.items() if k not in ('config_id', 'frozen_sha256')})
        write(out / 'plan.json', plan)
        for file, entry, _, rows in panels:
            target = out / 'evaluations' / identity(dict(config_id=plan['config_id'], kind=entry['kind']))
            cached = dict(entry, baseline=True, additional_steps=0, seed=seed, win_gain=None, wall_seconds=0., actual_new_games=0,
                          new_games_in_final_attempt=0, reused_cells_in_final_attempt=len(rows),
                          reused_from_sibling_games=0, reused_from_run=str(source), reused_from_evaluation=str(file.parent),
                          reuse_basis='Exact selected checkpoint; unchanged native evaluation conditions')
            write(target / 'completion.json', cached); write(out / 'candidates' / f'{target.name}.json', cached)
            for row in rows:
                key = {k: row[k] for k in ('opponent', 'seed', 'seat', 'action_seed')}
                write(target / 'cells' / f'{identity(key)}.json', row)
        write(out / 'baseline_checkpoint.json', dict(path=str(checkpoint)))
        write(out / 'parent_verification.json', dict(before=before, after=learner_state(restored),
              state_exact=True, byte_identical=True, initial_native_behavior_preserved=True,
              prepared_sha256=sha(prepared), source_sha256=sha(checkpoint / 'learner.zip'),
              comparator_wins=plan['qualified_parent_full_wins'], comparator_games=672,
              baseline_unique_games=672, baseline_panel_entries=800, new_baseline_games=0,
              all54_qualification=str(qualification), all54_wins=plan['qualified_parent_all54_wins']))
        reservation = ROOT / f'runs/training_reservation_{band}.json'
        with reservation.open('x') as file:
            import json
            json.dump(dict(run=str(out), start=band, stop_exclusive=band + 1000000), file)
        write(out / 'started.json', dict(time=time.time(), status='prepared_not_launched'))
        write(out / 'goal_control.json', dict(status='prepared', goal_status='active', publication=False))
        check_plan(original); check_plan(plan)
        print(dict(prepared=True, checkpoint=str(checkpoint), state_exact=True, new_baseline_games=0,
                   additional_steps=plan['maximum_steps'], wins21=plan['qualified_parent_full_wins'],
                   wins54=plan['qualified_parent_all54_wins']))
    finally:
        for lock in locks: lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'qualification', 'out'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--role', choices=['parent', 'candidate'], required=True)
    parser.add_argument('--seed', type=int, required=True); parser.add_argument('--band', type=int, required=True)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.qualification.resolve(), args.role, args.out.resolve(), args.seed, args.band)
