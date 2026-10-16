"""Prepare short continued PPO from one native-qualified early parent."""
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
from tools.prepare_aim_current_anchor import choose_full
from tools.rear_async_cpu import ROOT, check_plan


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
        qualification = ROOT / 'runs/aim_cpu_early59_complete_qualification_20261014'
        qualification_plan = read(qualification / 'plan.json'); check_plan(qualification_plan)
        qualified = read(qualification / 'QUALIFICATION_REVIEW.json')
        if not qualified['all672_paired_cells_reviewed'] or not qualified['all21_foe_groups_reviewed'] or qualified['selected_peak']['wins'] != 508:
            raise ValueError('Early parent is not fully qualified')
        if read(qualification / 'evaluation_complete.json')['status'] != 'queue_drained':
            raise ValueError('Qualification is not terminal')
        if original['ppo']['learning_rate'] != 5e-6:
            raise ValueError('Expected existing lower-rate learner configuration')
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
        entries.extend((f, read(f)) for f in (qualification / 'evaluations').glob('*/completion.json') if read(f)['kind'] == 'development' and not read(f)['baseline'])
        chosen = choose_full([entry for _, entry in entries])
        checkpoint = Path(chosen['checkpoint']).resolve(); metadata = verify(checkpoint)
        if (str(checkpoint), chosen['summary']['wins']) != (qualified['selected_peak']['path'], qualified['selected_peak']['wins']) or chosen['checkpoint_id'] != metadata['checkpoint_id']:
            raise ValueError('Reviewed qualified parent identity differs')
        parent = checkpoint / 'learner.zip'
        with zipfile.ZipFile(parent) as archive:
            serialized = json.loads(archive.read('data'))
            before = dict(policy=hashlib.sha256(archive.read('policy.pth')).hexdigest(), optimizer=hashlib.sha256(archive.read('policy.optimizer.pth')).hexdigest(), decisions=serialized['num_timesteps'], updates=serialized['_n_updates'])
        if sha(parent) != qualified['selected_peak']['zip_sha256'] or before['policy'] != qualified['selected_peak']['policy_sha256'] or before['optimizer'] != qualified['selected_peak']['optimizer_sha256']:
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
            matches = [(f, e) for f, e in entries if e['checkpoint_id'] == metadata['checkpoint_id'] and not e['baseline'] and e['kind'] == kind]
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
        seeds, bands = [47120, 47121], [3060400000, 3070400000]
        for file in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            reservation = read(file)
            if any(band < reservation['stop_exclusive'] and band + 1000000 > reservation['start'] for band in bands):
                raise ValueError('Reserved training band: ' + str(file))
        out.mkdir(parents=True)
        wins = chosen['summary']['wins']; prepared = out / f'qualified{wins}_parent.zip'; shutil.copyfile(parent, prepared)
        if sha(prepared) != sha(parent): raise ValueError('Copied learner bytes differ')
        runner = out / 'aim_cpu_runner.py'; code = (source / 'aim_cpu_runner.py').read_text()
        old = 'from tools.aim_cpu_head_rate_reporting import update'
        if code.count(old) != 1: raise ValueError('Reporter boundary differs')
        code = code.replace(old, 'from tools.aim_cpu_selected_head_reporting import update')
        compile(code, str(runner), 'exec'); runner.write_text(code)
        settings = dict(minimum_steps=1966080, maximum_steps=3932160, checkpoint_steps=655360, monitor_steps=655360, maximum_update_epochs_per_minimum=384, maximum_update_epochs_per_cap=768, maximum_update_epochs_per_checkpoint=128)
        assert settings['minimum_steps'] % settings['checkpoint_steps'] == 0 and settings['maximum_steps'] % settings['checkpoint_steps'] == 0
        assert settings['minimum_steps'] // settings['monitor_steps'] == 3 and settings['monitor_steps'] == settings['checkpoint_steps']
        budget = out / 'SHORT_CONTINUATION_BUDGET.json'; write(budget, dict(settings, reason='Qualifiedearly508 at1310720 beatslate495 at3932160;shorteradaptivecontinuation,notcausaloptimalbudgetclaim.'))
        expected_monitor = out / 'baseline_monitor_expected.json'; write(expected_monitor, monitor)
        plan = copy.deepcopy(original)
        plan.update(parent=str(prepared.relative_to(ROOT)), parent_source=str(parent), source_run=str(source), analysis_run=str(qualification), comparison_control_run=str(source), selected_parent_checkpoint=str(checkpoint), selected_parent_assessment=qualified['selected_peak']['assessment'], seeds=seeds, training_bands=bands, baseline_monitor_expected=str(expected_monitor), budget_spec_repository_relative=str(budget.relative_to(ROOT.parent)), qualified_parent_full_wins=wins, qualified_parent_full_games=672, scope=f'Short continuation from earlyprovisionalnative{wins}/672;original500 KLanchor retained.', parent_choice='Highest complete native672 checkpoint; preserve all original and specialist models, not the latest or monitor peak.', comparison_scope='Adaptiveadvance504to508 andshorterbudget/cadence/freshtrainingdraws;allrewards/PPO/mask/original500reference/roster/nativeconditions unchanged. Notcausalbudgetbenchmark/unseenvalidation.', historical_reference_fields='Inherited reference_* fields preserve league history; active adaptive comparator is selected_parent_checkpoint.')
        plan.update(settings)
        plan['intervention'] = dict(factor='qualified_starting_parent_and_shorter_budget', parent_checkpoint=metadata['checkpoint_id'], actor_features_trainable=False, unchanged='Rewards/43inputs/teacher/prior/48foes/probabilities/quarter0.5/PPO/LR5e-6/actorfeaturemask/original500KLanchor/nativephysics/guns/verdict/evaluationconditions', minimum_steps=1966080, maximum_steps=3932160, monitor_steps=655360)
        # The selected learner already stores the original anchor hash; reanchoring would be a second intervention.
        assert plan['reference_archive'] == original['reference_archive'] and plan['reference_policy_sha256'] == original['reference_policy_sha256']
        inputs = [prepared, runner, budget, expected_monitor, Path(__file__).resolve(), ROOT / 'tools/aim_cpu_selected_head_reporting.py']
        inputs.extend(source / name for name in ('plan.json', 'CYCLE_LOG_AUDIT.json', 'TWO_REPEAT_REVIEW.json', 'REPEAT_RESULTS.md', 'S47110_COMPLETED_REVIEW.json', 'S47111_COMPLETED_REVIEW.json', 'NEXT_EARLY59_QUALIFICATION_PLAN.md', 'REVIEW.json', 'TEST_EVIDENCE.json', 'STARTUP_EVIDENCE.json', 'training_complete.json', 'evaluation_complete.json'))
        inputs.append(Path(qualified['selected_peak']['assessment']))
        inputs.extend(source / name for name in audit['read_files'])
        for name, digest in qualification_plan['frozen_sha256'].items():
            if name in plan['frozen_sha256'] and plan['frozen_sha256'][name] != digest: raise ValueError('Conflicting inherited qualified input: ' + name)
            plan['frozen_sha256'][name] = digest
        inputs.extend(qualification / name for name in ('plan.json', 'QUALIFICATION_REVIEW.json', 'QUALIFICATION_REPORT.md', 'QUALIFICATION_SPEC.json', 'NEXT_SHORT508_PLAN.md', 'PREPARATION_EVIDENCE.json', 'REVIEW.json', 'STARTUP_EVIDENCE.json', 'evaluation_complete.json'))
        inputs.extend(ROOT / name for name in qualified['read_files'])
        for file, entry in entries:
            if entry['kind'] == 'development' and not entry['baseline']:
                used = Path(entry['checkpoint']); used_meta = verify(used)
                inputs.extend([used / 'READY.json', *(used / name for name in used_meta['files'])])
        for file, _, _, files in panels: inputs.extend([file, *files])
        for file in inputs:
            name, digest = str(file.relative_to(ROOT)), sha(file)
            if name in plan['frozen_sha256'] and plan['frozen_sha256'][name] != digest: raise ValueError('Frozen input changed: ' + name)
            plan['frozen_sha256'][name] = digest
        plan['config_id'] = identity({k: v for k, v in plan.items() if k not in ('config_id', 'frozen_sha256')}); write(out / 'plan.json', plan)
        for file, entry, rows, _ in panels:
            folder = out / 'evaluations' / identity(dict(config_id=plan['config_id'], parent=metadata['checkpoint_id'], kind=entry['kind']))
            reused = dict(entry, seed=seeds[0], additional_steps=0, baseline=True, win_gain=None, candidate=False, improvement_confirmed=False, status=f'reused_byte_identical_qualified{wins}_initial_learner', actual_new_games=0, new_games_in_final_attempt=0, reused_cells_in_final_attempt=len(rows), reused_from_sibling_games=0, wall_seconds=0., reused_from_run=str(source), reused_from_evaluation=str(file.parent), reuse_basis='Selected learner ZIP is byte-identical; model/export/conditions/samplers preserved; mask and stored original500 KLanchor unchanged.')
            write(folder / 'completion.json', reused); write(out / 'candidates' / f'{folder.name}.json', reused)
            for row in rows: write(folder / 'cells' / f"{identity({k: row[k] for k in ('opponent', 'seed', 'seat', 'action_seed')})}.json", row)
        write(out / 'baseline_checkpoint.json', dict(path=str(checkpoint)))
        write(out / 'parent_verification.json', dict(before=before, after=before, restored=before, state_exact=True, source_sha256=sha(parent), prepared_sha256=sha(prepared), byte_identical=True, byte_identical_to_qualified_parent=True, state_basis='Byte-identical serialized copy; no redundant model-load test.', initial_native_behavior_preserved=True, policy_behavior_unchanged=True, comparator_wins=wins, comparator_games=672, baseline_unique_games=672, baseline_panel_entries=800, new_baseline_games=0, original_kl_anchor_preserved=True, reference_policy_sha256=plan['reference_policy_sha256'], actor_features_trainable=False, source_actor_feature_sha256=plan['source_actor_feature_sha256'], qualified_checkpoint=str(checkpoint)))
        for band in bands:
            with (ROOT / f'runs/training_reservation_{band}.json').open('x') as file: json.dump(dict(run=str(out), start=band, stop_exclusive=band + 1000000), file)
        write(out / 'started.json', dict(time=time.time(), status='prepared_not_launched'))
        write(out / 'goal_control.json', dict(status='prepared_review_pending', goal_status='active', automatic_learning=True, publication=False, next_action='Onecumulative read-onlyreview,then shortnative508CPU PPO continuation.'))
        check_plan(original); check_plan(plan)
        print(json.dumps(dict(prepared=True, comparator_wins=wins, hard_monitor_wins=next(e['summary']['wins'] for _, e, _, _ in panels if e['kind'] == 'monitor'), byte_identical=True, original_kl_anchor_preserved=True, prefix_steps=before['decisions'], prefix_updates=before['updates'], new_baseline_games=0, frozen_files=len(plan['frozen_sha256']))))
    finally:
        for lock in locks: lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); prepare(args.source.resolve(), args.out.resolve())
