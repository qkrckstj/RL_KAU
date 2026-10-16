"""Prepare explicit current-parent reference rebinding with learned payloads intact."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import shutil
import time
import zipfile
from pathlib import Path
from experiments.rear_async.storage import identity, read, sha, verify, write
from experiments.aim_async.actor_head import archive_actor_feature_digest
from tools.prepare_aim_actor_head import short_budget
from experiments.aim_async.reference_archive import rebind_reference_archive
from tools.rear_async_cpu import ROOT, check_plan


def choose_full(entries):
    full = [entry for entry in entries if entry['kind'] == 'development']
    if any(entry['summary']['games'] != 672 for entry in full):
        raise ValueError('Incomplete native full panel')
    baseline = [entry for entry in full if entry['baseline']]
    if len(baseline) != 1:
        raise ValueError('Ambiguous native comparator')
    better = [entry for entry in full if not entry['baseline'] and entry['summary']['wins'] > baseline[0]['summary']['wins']]
    return max(better, key=lambda entry: (entry['summary']['wins'], entry['summary']['damage_dealt'] - entry['summary']['damage_received'], -entry['additional_steps'])) if better else baseline[0]


def prepare(source, out):
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Use a fresh run namespace')
    import psutil
    for process in psutil.process_iter(['cmdline', 'status']):
        args = process.info['cmdline'] or []
        if process.info['status'] != psutil.STATUS_ZOMBIE and any(Path(a).name == 'aim_cpu_runner.py' or a == 'tools.rear_async_cpu' for a in args) and any(a in args for a in ('launch', 'evaluate')):
            raise RuntimeError('Existing native trainer/evaluator: ' + str(process.pid))
    if psutil.virtual_memory().available < 10 * 2**30:
        raise MemoryError('Need10GiB available')
    locks = []
    try:
        for name in ('launch.lock', 'training.lock', 'evaluation.lock'):
            lock = (source / name).open('a'); locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (source / 'training_complete.json').exists() or (source / 'training_failure.json').exists() or read(source / 'evaluation_complete.json')['status'] != 'queue_drained':
            raise ValueError('Source is not successfully terminal')
        for file in (source / 'queue').glob('*.json'):
            folder = source / 'evaluations' / read(file)['job_id']
            if not (folder / 'completion.json').exists() or (folder / 'blocked.json').exists():
                raise ValueError('Incomplete or blocked source queue')
        original = read(source / 'plan.json'); check_plan(original)
        review = read(source / 'TWO_REPEAT_REVIEW.json'); audit = read(source / 'CYCLE_LOG_AUDIT.json')
        if review['status'] != 'two_repeats_complete_analyzed' or set(audit['training']) != {str(s) for s in original['seeds']}:
            raise ValueError('Source repeats not fully analyzed')
        for seed in original['seeds']:
            recorded = audit['training'][str(seed)]
            if not (source / f's{seed}_learning_complete.json').exists() or recorded['all_episodes']['games'] != read(source / f's{seed}_result.json')['training_games'] or recorded['nonfinite_ppo_fields']:
                raise ValueError('Source audit differs or contains nonfinite metrics')
        if original['actor_features_trainable'] is not False or original['reference_kl_coefficient'] != .1 or original['ppo']['gae_lambda'] != .99 or (original['processes'], original['workers']) != (8, 32):
            raise ValueError('Expected existing actor-feature-freeze configuration')
        for key in ('minimum_steps', 'maximum_steps', 'checkpoint_steps', 'monitor_steps'):
            if original[key] != short_budget()[key]:
                raise ValueError('Source budget differs: ' + key)
        entries = [(f, read(f)) for f in (source / 'evaluations').glob('*/completion.json')]
        chosen = choose_full([entry for _, entry in entries])
        checkpoint = Path(chosen['checkpoint']).resolve(); metadata = verify(checkpoint)
        if (str(checkpoint), chosen['summary']['wins']) != (review['selected_peak']['path'], review['selected_peak']['wins']) or chosen['checkpoint_id'] != metadata['checkpoint_id']:
            raise ValueError('Reviewed qualified parent identity differs')
        parent = checkpoint / 'learner.zip'
        with zipfile.ZipFile(parent) as archive:
            serialized = json.loads(archive.read('data'))
            before = dict(policy=hashlib.sha256(archive.read('policy.pth')).hexdigest(), optimizer=hashlib.sha256(archive.read('policy.optimizer.pth')).hexdigest(), decisions=serialized['num_timesteps'], updates=serialized['_n_updates'])
        if sha(parent) != review['selected_peak']['zip_sha256'] or before['policy'] != review['selected_peak']['policy_sha256'] or before['optimizer'] != review['selected_peak']['optimizer_sha256']:
            raise ValueError('Selected learner payload differs from completed review')
        for key in ('learning_rate', 'n_steps', 'n_epochs', 'batch_size', 'gamma', 'gae_lambda', 'target_kl', 'ent_coef', 'max_grad_norm'):
            if serialized[key] != original['ppo'][key]:
                raise ValueError('Selected learner PPO setting differs: ' + key)
        if serialized['num_timesteps'] != metadata['total_steps'] or serialized['n_envs'] != 32 or serialized['reference_policy_hash'] != original['reference_policy_sha256'] or serialized['reference_coefficient'] != original['reference_kl_coefficient']:
            raise ValueError('Selected counters or original KL anchor differ')
        exported = read(checkpoint / 'greedy/policy_net.json')
        if exported['observation_dim'] != 43 or exported['prior_bias'] != original['prior_bias'] or exported['teacher'] != original['teacher'] or serialized['policy_kwargs']['prior_bias'] != original['prior_bias']:
            raise ValueError('Selected feature/teacher/prior contract differs')
        if archive_actor_feature_digest(parent) != original['source_actor_feature_sha256']:
            raise ValueError('Frozen actor feature identity differs')
        from experiments.aim_async.evaluate import profile
        panels = []
        for kind in ('monitor', 'development'):
            matches = [(f, e) for f, e in entries if e['checkpoint_id'] == metadata['checkpoint_id'] and e['kind'] == kind]
            if len(matches) != 1: raise ValueError('Ambiguous qualified panel')
            file, entry = matches[0]; files = sorted(file.parent.glob('cells/*.json')); rows = [read(f) for f in files]
            replicas = original['monitor_action_seeds'] if kind == 'monitor' else [4900, 4901]
            expected = {(foe['id'], original[kind + '_band'] + i, seat, action) for foe in original[kind + '_opponents'] for i in range(original[kind + '_ics']) for seat in ('red', 'blue') for action in replicas}
            keys = {tuple(row[k] for k in ('opponent', 'seed', 'seat', 'action_seed')) for row in rows}
            if len(rows) != len(expected) or keys != expected: raise ValueError('Incomplete qualified panel')
            measured = profile(rows)
            if measured.keys() != entry['summary'].keys() or any(measured[k] != v and not (measured[k] is not None and v is not None and math.isclose(measured[k], v, rel_tol=0., abs_tol=1e-12)) for k, v in entry['summary'].items()):
                raise ValueError('Qualified panel summary differs')
            panels.append((file, entry, rows, files))
        full = next(rows for _, e, rows, _ in panels if e['kind'] == 'development'); monitor = next(rows for _, e, rows, _ in panels if e['kind'] == 'monitor')
        index = lambda rows: {tuple(row[k] for k in ('opponent', 'seed', 'seat', 'action_seed')): row for row in rows}
        if len(full) != 672 or len(monitor) != 128 or any(index(full).get(k) != row for k, row in index(monitor).items()):
            raise ValueError('Qualified monitor is not an exact full subset')
        seeds, bands = [47100, 47101], [3020400000, 3030400000]
        for file in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            reservation = read(file)
            if any(band < reservation['stop_exclusive'] and band + 1000000 > reservation['start'] for band in bands):
                raise ValueError('Reserved training band: ' + str(file))
        out.mkdir(parents=True)
        wins = chosen['summary']['wins']; prepared = out / f'qualified{wins}_reanchored_parent.zip'
        migration = rebind_reference_archive(parent, prepared, original['reference_policy_sha256'])
        if (migration['policy_sha256'], migration['optimizer_sha256'], migration['source_num_timesteps'], migration['source_n_updates']) != (before['policy'], before['optimizer'], before['decisions'], before['updates']): raise ValueError('Reference rebinding changed learning state')
        write(out / 'REFERENCE_MIGRATION.json', migration)
        runner = out / 'aim_cpu_runner.py'; code = (source / 'aim_cpu_runner.py').read_text()
        old = 'from tools.aim_cpu_selected_head_reporting import update'
        if code.count(old) != 1: raise ValueError('Reporter boundary differs')
        code = code.replace(old, 'from tools.aim_cpu_current_anchor_reporting import update')
        compile(code, str(runner), 'exec'); runner.write_text(code)
        budget = out / 'SHORT_CONTINUATION_BUDGET.json'; write(budget, dict(short_budget(), reason='Unchanged actor-feature-freeze continuation; only the fixed KL anchor moves from original500 to current504.'))
        expected_monitor = out / 'baseline_monitor_expected.json'; write(expected_monitor, monitor)
        plan = copy.deepcopy(original)
        plan.update(parent=str(prepared.relative_to(ROOT)), parent_source=str(parent), source_run=str(source), analysis_run=str(source), comparison_control_run=str(source), selected_parent_checkpoint=str(checkpoint), selected_parent_assessment=review['selected_peak']['assessment'], seeds=seeds, training_bands=bands, baseline_monitor_expected=str(expected_monitor), budget_spec_repository_relative=str(budget.relative_to(ROOT.parent)), qualified_parent_full_wins=wins, qualified_parent_full_games=672, scope=f'Fixed actor-feature continuation from provisional native{wins}/672; fixedKLanchor is currentqualifiedparent.', parent_choice='Highest complete native672 checkpoint; preserve all original and specialist models, not the latest or monitor peak.', comparison_scope='Same504startingNN/Adam/counters/rewards/PPO/mask/nativeconditions; onlyfixedanchor500to504 andfreshtrainingdraws change. Consumeddevelopment isnot unseenvalidation.', historical_reference_fields='Inherited reference_* fields preserve league history; active adaptive comparator is selected_parent_checkpoint.')
        plan['intervention'] = dict(factor='fixed_reference_actor', control_policy_sha256=original['reference_policy_sha256'], value_policy_sha256=before['policy'], parent_checkpoint=metadata['checkpoint_id'], actor_features_trainable=False, unchanged='Starting504NN/Adam/counters/rewards/43inputs/teacher/prior/48foes/probabilities/quarter0.5/PPO/actorfeaturemask/nativephysics/guns/verdict/evaluation/budget')
        # Rebinding is an explicit single new-study factor; learned payloads and resume guards remain intact.
        plan.update(previous_reference_archive=original['reference_archive'], previous_reference_policy_sha256=original['reference_policy_sha256'], reference_archive=str(prepared.relative_to(ROOT)), reference_policy_sha256=before['policy'])
        assert migration['new_reference_policy_sha256'] == plan['reference_policy_sha256']
        inputs = [prepared, runner, budget, expected_monitor, Path(__file__).resolve(), ROOT / 'tools/aim_cpu_current_anchor_reporting.py', ROOT / 'experiments/aim_async/reference_archive.py', ROOT / 'tests/test_aim_reference_rebind.py', out / 'REFERENCE_MIGRATION.json']
        inputs.extend(source / name for name in ('plan.json', 'CYCLE_LOG_AUDIT.json', 'TWO_REPEAT_REVIEW.json', 'REPEAT_RESULTS.md', 'S47090_COMPLETED_REVIEW.json', 'S47091_COMPLETED_REVIEW.json', 'NEXT_CURRENT_ANCHOR_PLAN.md', 'REVIEW.json', 'PREPARATION_EVIDENCE.json', 'STARTUP_EVIDENCE.json', 'training_complete.json', 'evaluation_complete.json'))
        inputs.append(Path(review['selected_peak']['assessment']))
        inputs.extend(source / name for name in audit['read_files'])
        for file, entry in entries:
            if entry['kind'] == 'development' and not entry['baseline']:
                used = Path(entry['checkpoint']); used_meta = verify(used)
                inputs.extend([used / 'READY.json', *(used / name for name in used_meta['files'])])
        for file, _, _, files in panels: inputs.extend([file, *files])
        plan['frozen_sha256'].update({str(file.relative_to(ROOT)): sha(file) for file in inputs})
        plan['config_id'] = identity({k: v for k, v in plan.items() if k not in ('config_id', 'frozen_sha256')}); write(out / 'plan.json', plan)
        for file, entry, rows, _ in panels:
            folder = out / 'evaluations' / identity(dict(config_id=plan['config_id'], parent=metadata['checkpoint_id'], kind=entry['kind']))
            reused = dict(entry, seed=seeds[0], additional_steps=0, baseline=True, win_gain=None, candidate=False, improvement_confirmed=False, status=f'reused_exact_policy_payload_qualified{wins}_initial_learner', actual_new_games=0, new_games_in_final_attempt=0, reused_cells_in_final_attempt=len(rows), reused_from_sibling_games=0, wall_seconds=0., reused_from_run=str(source), reused_from_evaluation=str(file.parent), reuse_basis='Allnon-dataZIPpayloads/NN/export/nativeforward/conditions/samplers matchselectedparent. Onlyreferencehash/pathmetadata change; nativeinference ignores the optimizationreference.')
            write(folder / 'completion.json', reused); write(out / 'candidates' / f'{folder.name}.json', reused)
            for row in rows: write(folder / 'cells' / f"{identity({k: row[k] for k in ('opponent', 'seed', 'seat', 'action_seed')})}.json", row)
        write(out / 'baseline_checkpoint.json', dict(path=str(checkpoint)))
        write(out / 'parent_verification.json', dict(before=before, after=before, restored=before, state_exact=True, source_sha256=sha(parent), prepared_sha256=sha(prepared), byte_identical=False, byte_identical_to_qualified_parent=False, non_data_payloads_exact=True, state_basis='Alllearnedpayloads/NN/initializedAdam/counters andnon-reference metadata exact; referencehash/path only intentionally rebound.', initial_native_behavior_preserved=True, policy_behavior_unchanged=True, comparator_wins=wins, comparator_games=672, baseline_unique_games=672, baseline_panel_entries=800, new_baseline_games=0, original_kl_anchor_preserved=False, original_kl_anchor_archived=True, reference_rebinding_explicit=True, reference_policy_sha256=plan['reference_policy_sha256'], actor_features_trainable=False, source_actor_feature_sha256=plan['source_actor_feature_sha256'], qualified_checkpoint=str(checkpoint)))
        for band in bands:
            with (ROOT / f'runs/training_reservation_{band}.json').open('x') as file: json.dump(dict(run=str(out), start=band, stop_exclusive=band + 1000000), file)
        write(out / 'started.json', dict(time=time.time(), status='prepared_not_launched'))
        write(out / 'goal_control.json', dict(status='prepared_review_pending', goal_status='active', automatic_learning=True, publication=False, next_action='One cumulative read-only review, then current-parent reference CPU PPO continuation.'))
        check_plan(original); check_plan(plan)
        print(json.dumps(dict(prepared=True, comparator_wins=wins, hard_monitor_wins=next(e['summary']['wins'] for _, e, _, _ in panels if e['kind'] == 'monitor'), non_data_payloads_exact=True, reference_rebinding_explicit=True, old_reference_policy_sha256=original['reference_policy_sha256'], new_reference_policy_sha256=before['policy'], prefix_steps=before['decisions'], prefix_updates=before['updates'], new_baseline_games=0, frozen_files=len(plan['frozen_sha256']))))
    finally:
        for lock in locks: lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); prepare(args.source.resolve(), args.out.resolve())
