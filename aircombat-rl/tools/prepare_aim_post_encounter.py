"""Prepare one native post-encounter curriculum without editing frozen sources."""
import argparse
import copy
import fcntl
import math
from pathlib import Path
import shutil
import time

from experiments.rear_async.storage import identity, read, sha, verify, write
from tools.aim_known_qualification import index_panel


def assert_profile(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError('Summary fields differ')
    for key, value in expected.items():
        if actual[key] != value and not (actual[key] is not None and value is not None and
                                        math.isclose(actual[key], value, rel_tol=0, abs_tol=1e-12)):
            raise ValueError('Summary differs: ' + key)


def completed_panel(folder, checkpoint, plan):
    """Validate cached all-opponent evidence against its exact policy and conditions."""
    from experiments.aim_async.evaluate import profile
    completion = read(folder / 'completion.json')
    manifest = read(folder / 'manifest.json')
    historical_failure = []
    failure = folder / 'failure.json'
    if completion['status'] != 'complete':
        raise ValueError('Qualification is not successfully complete')
    if failure.exists():
        runtime = folder / 'runtime.json'
        if (not runtime.exists() or read(runtime)['status'] != 'complete' or
                (folder / 'completion.json').stat().st_mtime_ns <= failure.stat().st_mtime_ns):
            raise ValueError('Qualification has an unresolved failure')
        historical_failure = [failure, runtime]
    contract = {k: v for k, v in manifest.items() if k not in ('contract_id', 'started')}
    if identity(contract) != manifest['contract_id']:
        raise ValueError('Qualification contract changed')
    roles = [role for role in ('parent', 'candidate')
             if Path(manifest[role]).resolve() == checkpoint.resolve()]
    if len(roles) != 1:
        raise ValueError('Qualification belongs to another checkpoint')
    role = roles[0]
    if (manifest['opponents'] != plan['opponents'] or
            manifest['band'] != plan['development_band'] or
            manifest['ics'] != plan['development_ics'] or manifest['samplers'] != [4900, 4901]):
        raise ValueError('Qualification conditions differ')
    verify(checkpoint)
    ready = str((checkpoint / 'READY.json').resolve())
    if manifest['source_sha256'].get(ready) != sha(checkpoint / 'READY.json'):
        raise ValueError('Qualification checkpoint publication changed')
    for name, digest in manifest['source_sha256'].items():
        if sha(Path(name)) != digest:
            raise ValueError('Qualification source changed: ' + name)
    files = sorted((folder / role / 'cells').glob('*.json'))
    rows = []
    for file in files:
        record = read(file)
        if (record['contract_id'] != manifest['contract_id'] or record['role'] != role or
                record['origin'] not in ('new', 'reused') or identity(record['row']) != record['row_sha256']):
            raise ValueError('Qualification cell changed')
        rows.append(record['row'])
    index_panel(rows, [o['id'] for o in plan['opponents']], manifest['band'], manifest['ics'], [4900, 4901])
    assert_profile(profile(rows), completion['policies'][role]['summary'])
    return rows, [folder / 'manifest.json', folder / 'completion.json', *files,
                  *map(Path, manifest['source_sha256']), *historical_failure]


def continuation_plan(original, parent, source, seed, band, metadata, wins21, wins54,
                      checkpoint, assessment, monitor_expected):
    plan = copy.deepcopy(original)
    plan.update(parent=str(parent), parent_source=str(parent), source_run=str(source),
                seeds=[seed], training_bands=[band], source_actor_feature_sha256=metadata['actor_feature_sha256'],
                qualified_parent_full_wins=wins21, qualified_parent_full_games=672,
                qualified_parent_all54_wins=wins54, qualified_parent_all54_games=1728,
                selected_parent_checkpoint=str(checkpoint), selected_parent_assessment=str(assessment),
                baseline_monitor_expected=str(monitor_expected),
                scope='Same-setting long continuation against the unchanged54 known opponents',
                intervention=dict(factor='continuation_only',
                                  unchanged='NN/Adam/counters at initialization; PPO/rewards/reference/roster/weights/curriculum/controller/physics'),
                comparison_scope='Consumed known-opponent development; not unseen validation')
    return plan


def scale_budget(plan, multiplier):
    if type(multiplier) is not int or multiplier < 1:
        raise ValueError('Budget multiplier must be a positive integer')
    result = copy.deepcopy(plan)
    for key in ('minimum_steps', 'maximum_steps', 'monitor_steps', 'development_steps',
                'maximum_update_epochs_per_minimum', 'maximum_update_epochs_per_cap'):
        result[key] *= multiplier
    result['budget_policy'] = f"Fixed{result['maximum_steps']:,}decision segment; no intermediate plateau stopping; asynchronous evaluations"
    return result


def prepare(source, qualification, role, out, seed, band, budget_multiplier=1):
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
        old = 'from tools.aim_cpu_known_long_reporting import update'
        new = 'from tools.aim_known_continuation_reporting import update'
        focused = 'from tools.aim_known_mutual_focus_reporting import update'
        if sum(code.count(line) for line in (old, new, focused)) != 1:
            raise ValueError('Unknown or ambiguous runner reporter boundary')
        if code.count(old) == 1: code = code.replace(old, new)
        elif code.count(focused) == 1: code = code.replace(focused, new)
        from tools.aim_post_encounter import transform_plan, transform_runner
        code = transform_runner(code)
        reporter = 'from tools.aim_known_continuation_reporting import update'
        if code.count(reporter) != 1: raise ValueError('Unknown reporter boundary')
        code = code.replace(reporter, 'from tools.aim_post_encounter_reporting import update')
        runner = out / 'aim_cpu_runner.py'; compile(code, str(runner), 'exec'); runner.write_text(code)
        monitor_expected = out / 'baseline_monitor_expected.json'
        write(monitor_expected, panels[0][3])
        plan = continuation_plan(original, prepared.relative_to(ROOT), source, seed, band, metadata,
                                 panels[1][1]['summary']['wins'], profile(rows54)['wins'],
                                 checkpoint, qualification / 'completion.json', monitor_expected)
        plan = scale_budget(plan, budget_multiplier)
        plan = transform_plan(plan, metadata['sampled_specs']['4900'])
        plan.update(parent_source=str((checkpoint / 'learner.zip').relative_to(ROOT)),
                    baseline_all54_qualification=str(qualification), baseline_all54_checkpoint=str(checkpoint),
                    budget_spec_repository_relative='docs/superpowers/plans/2026-10-15-native-post-encounter.md')
        files = [Path(__file__), ROOT.parent / 'docs/superpowers/plans/2026-10-15-native-post-encounter.md', ROOT / 'tools/aim_post_encounter.py', ROOT / 'tests/test_post_encounter.py',
                 ROOT / 'experiments/aim_async/post_encounter.py',
                 ROOT / 'tools/aim_post_encounter_reporting.py', ROOT / 'tests/test_post_encounter_reporting.py',
                 ROOT / 'tools/aim_known_continuation_reporting.py',
                 source / 'plan.json', source / 'CYCLE_LOG_AUDIT.json', checkpoint / 'READY.json',
                 prepared, runner, monitor_expected, *evidence_files]
        for file, _, paths, _ in panels: files.extend([file, *paths])
        plan['frozen_sha256'].update({__import__('os').path.relpath(p.resolve(), ROOT): sha(p) for p in files})
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
    parser.add_argument('--budget-multiplier', type=int, default=1)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.qualification.resolve(), args.role, args.out.resolve(), args.seed, args.band,
            args.budget_multiplier)
