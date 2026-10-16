"""Lower a constant PPO update rate while retaining the qualified learner state."""
import argparse
import copy
import fcntl
import json
import math
import shutil
import time
from pathlib import Path

from experiments.rear_async.storage import identity, read, sha, verify, write
from tools.aim_prior_bias import learner_state
from tools.rear_async_cpu import ROOT, check_plan


def change_learning_rate(model, target, expected_source=3e-5):
    target = float(target)
    if not math.isfinite(target) or not 0. < target < expected_source:
        raise ValueError('Use a lower positive finite learning rate')
    if model.learning_rate != expected_source or any(
            model.lr_schedule(p) != expected_source for p in (0., .5, 1.)):
        raise ValueError('Unexpected source learning-rate scalar or schedule')
    before = learner_state(model)
    model.learning_rate = target
    model._setup_lr_schedule()
    after = learner_state(model)
    if before != after:
        raise ValueError('Initial policy/Adam/counters changed')
    return dict(before=before, after=after, state_exact=True,
                source_rate=expected_source, target_rate=target,
                optimizer_rate_applied_by='SB3 PPO.train before its updates')


def prepare(source, out, rate=1e-5):
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a fresh output under runs')
    if not (source/'training_complete.json').exists() or (source/'training_failure.json').exists():
        raise RuntimeError('Source not successfully terminal')
    if read(source/'evaluation_complete.json')['status'] != 'queue_drained' or list((source/'evaluations').glob('*/blocked.json')):
        raise RuntimeError('Source evaluation incomplete or blocked')
    import psutil
    for process in psutil.process_iter(['cmdline', 'status']):
        args = process.info['cmdline'] or []
        if (process.info['status'] != psutil.STATUS_ZOMBIE
                and any(Path(a).name == 'aim_cpu_runner.py' or a == 'tools.rear_async_cpu' for a in args)
                and any(a in args for a in ('launch', 'evaluate'))):
            raise RuntimeError('Existing native trainer/evaluator: '+str(process.pid))
    if psutil.virtual_memory().available < 10*2**30:
        raise MemoryError('Need10GiB available')
    locks = []
    try:
        for name in ('launch.lock', 'training.lock', 'evaluation.lock'):
            lock = (source/name).open('a'); locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = read(source/'plan.json'); check_plan(original)
        review = read(source/'TWO_REPEAT_REVIEW.json'); audit = read(source/'CYCLE_LOG_AUDIT.json')
        if review['status'] != 'two_repeats_complete_analyzed' or set(audit['training']) != {str(s) for s in original['seeds']}:
            raise ValueError('Source repeats not completely analyzed')
        for seed in original['seeds']:
            trained = audit['training'][str(seed)]
            if trained['all_episodes']['games'] != read(source/f's{seed}_result.json')['training_games'] or trained['nonfinite_ppo_fields']:
                raise ValueError('Completed audit differs or nonfinite')
        if (original['relative_enemy_feedback'] != 'public_native_predictive_lead_cos_v1'
                or original['ppo']['learning_rate'] != 3e-5
                or original['learner_feature_dim'] != 43 or original['prior_bias'] != 6.
                or original['processes'] != 8 or original['workers'] != 32):
            raise ValueError('Expected completed predictive-enemy rate3e-5 control')
        entries = [(p, read(p)) for p in (source/'evaluations').glob('*/completion.json')]
        baseline = next(e for _, e in entries if e['baseline'] and e['kind'] == 'development')
        endpoints = [e for _, e in entries if not e['baseline'] and e['kind'] == 'development']
        if len(endpoints) != 2 or any(e['summary']['wins'] >= baseline['summary']['wins'] for e in endpoints):
            raise ValueError('Review improved/incomplete endpoints before changing the rate')
        checkpoint = Path(read(source/'baseline_checkpoint.json')['path']).resolve()
        metadata = verify(checkpoint)
        source_evidence = read(source/'parent_verification.json')
        parent = ROOT/original['parent']
        if (not source_evidence['state_exact'] or sha(parent) != source_evidence['prepared_sha256']
                or sha(parent) != sha(checkpoint/'learner.zip')):
            raise ValueError('Inherited qualified parent learner differs')
        from experiments.aim_async.evaluate import profile
        panels = []
        for kind in ('monitor', 'development'):
            matches = [(p, e) for p, e in entries if e['baseline'] and e['kind'] == kind]
            if len(matches) != 1:
                raise ValueError('Ambiguous baseline panel')
            p, entry = matches[0]
            if entry['checkpoint_id'] != metadata['checkpoint_id']:
                raise ValueError('Baseline checkpoint identity differs')
            files = sorted((p.parent/'cells').glob('*.json')); rows = [read(f) for f in files]
            samplers = original['monitor_action_seeds'] if kind == 'monitor' else [4900, 4901]
            expected = {(f['id'], original[kind+'_band']+i, seat, action)
                        for f in original[kind+'_opponents'] for i in range(original[kind+'_ics'])
                        for seat in ('red', 'blue') for action in samplers}
            keys = {tuple(r[k] for k in ('opponent', 'seed', 'seat', 'action_seed')) for r in rows}
            if len(rows) != len(expected) or keys != expected:
                raise ValueError('Incomplete or unpaired native baseline')
            measured = profile(rows)
            if measured.keys() != entry['summary'].keys() or any(
                    measured[k] != v and not (measured[k] is not None and v is not None
                    and math.isclose(measured[k], v, rel_tol=0., abs_tol=1e-12))
                    for k, v in entry['summary'].items()):
                raise ValueError('Native baseline profile differs')
            panels.append((p, entry, rows, files))
        full = next(rows for _, e, rows, _ in panels if e['kind'] == 'development')
        monitor = next(rows for _, e, rows, _ in panels if e['kind'] == 'monitor')
        indexed = lambda rows: {tuple(r[k] for k in ('opponent', 'seed', 'seat', 'action_seed')):r for r in rows}
        if len(full) != 672 or len(monitor) != 128 or any(indexed(full).get(k) != r for k, r in indexed(monitor).items()):
            raise ValueError('Monitor is not the exact native full-panel subset')
        seeds, bands = [46810, 46811], [2780400000, 2790400000]
        for path in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            reservation = read(path)
            if any(b < reservation['stop_exclusive'] and b+1000000 > reservation['start'] for b in bands):
                raise ValueError('Reserved training band: '+str(path))
        import torch
        from stable_baselines3 import PPO
        from experiments.aim_async.policy import AimPriorPolicy
        torch.set_num_threads(1); model = PPO.load(parent, device='cpu')
        if type(model.policy) is not AimPriorPolicy or model.observation_space.shape != (43,) or model.policy.prior_bias != 6. or model.n_envs != 32:
            raise ValueError('Parent policy/spaces/prior differ')
        if learner_state(model) != source_evidence['before'] or model.num_timesteps != metadata['total_steps']:
            raise ValueError('Inherited NN/Adam/counters differ')
        for key in ('n_steps', 'n_epochs', 'batch_size', 'gamma', 'gae_lambda', 'target_kl', 'ent_coef', 'max_grad_norm'):
            if getattr(model, key) != original['ppo'][key]:
                raise ValueError('Parent PPO setting differs: '+key)
        if model.clip_range(1.) != original['ppo']['clip_range']:
            raise ValueError('Parent clip schedule differs')
        evidence = change_learning_rate(model, rate)
        out.mkdir(parents=True); prepared = out/'lower_lr_parent.zip'; model.save(prepared)
        restored = PPO.load(prepared, device='cpu')
        if learner_state(restored) != evidence['before'] or any(restored.lr_schedule(p) != rate for p in (0., .5, 1.)):
            raise ValueError('Saved rate or original learner state changed')
        runner = out/'aim_cpu_runner.py'
        runner_code = (source/'aim_cpu_runner.py').read_text()
        old_report = 'from tools.aim_cpu_predictive_enemy_reporting import update'
        if runner_code.count(old_report) != 1:
            raise ValueError('Reporter import boundary differs')
        runner_code = runner_code.replace(old_report, 'from tools.aim_cpu_learning_rate_reporting import update')
        compile(runner_code, str(runner), 'exec'); runner.write_text(runner_code)
        plan = copy.deepcopy(original); plan['ppo']['learning_rate'] = rate
        plan.update(parent=str(prepared.relative_to(ROOT)), parent_source=str(parent), source_run=str(source),
                    seeds=seeds, training_bands=bands, scope='Fixed lower-rate PPO from qualified499reference.',
                    intervention=dict(factor='ppo.learning_rate', control=3e-5, value=rate,
                        unchanged='initialNN/initializedAdam/counters/43features/predictive reward/draw0.25/prior6/teacher/PPO except rate/48foes/quarter0.5/nativephysics/guns/verdict/evaluation'),
                    comparison_scope='Adaptive single-scalar update-rate study; initial policy identical; consumed672development,not unseen final.',
                    parent_choice='Preserved strongest completed672reference; provisional development choice')
        inputs = [prepared, runner, Path(__file__).resolve(), ROOT/'tests/test_aim_learning_rate.py',
                  source/'plan.json', source/'parent_verification.json', source/'CYCLE_LOG_AUDIT.json',
                  source/'TWO_REPEAT_REVIEW.json', source/'S46700_COMPLETED_REVIEW.json',
                  source/'S46701_COMPLETED_REVIEW.json', source/'REPEAT_RESULTS.md',
                  source/'LOWER_LR_NEXT_STUDY.md', source/'LOWER_LR_REPORT_CORRECTION.md',
                  ROOT/'tools/prepare_aim_learning_rate.py', ROOT/'tools/aim_cpu_learning_rate_reporting.py',
                  checkpoint/'READY.json']
        inputs.extend(checkpoint/name for name in metadata['files'])
        for p, _, _, files in panels:
            inputs += [p, *files]
        plan['frozen_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in inputs})
        plan['config_id'] = identity({k:v for k, v in plan.items() if k not in ('config_id', 'frozen_sha256')})
        write(out/'plan.json', plan)
        for p, entry, rows, _ in panels:
            folder = out/'evaluations'/identity(dict(config_id=plan['config_id'], parent=metadata['checkpoint_id'], kind=entry['kind']))
            reused = dict(entry, seed=seeds[0], additional_steps=0, baseline=True,
                status='reused_identical_initial_policy_lower_lr', actual_new_games=0,
                new_games_in_final_attempt=0, reused_cells_in_final_attempt=len(rows), wall_seconds=0.,
                reused_from_run=str(source), reused_from_evaluation=str(p.parent),
                reuse_basis='NN/initializedAdam/counters/43features/prior unchanged. Only future PPO update rate changes; initial native actor identical.',
                per_opponent={foe:profile([r for r in rows if r['opponent'] == foe]) for foe in sorted({r['opponent'] for r in rows})})
            write(folder/'completion.json', reused); write(out/'candidates'/f'{folder.name}.json', reused)
            for row in rows:
                write(folder/'cells'/f"{identity({k:row[k] for k in ('opponent', 'seed', 'seat', 'action_seed')})}.json", row)
        write(out/'baseline_checkpoint.json', dict(path=str(checkpoint)))
        write(out/'parent_verification.json', dict(evidence, restored=learner_state(restored),
            source_sha256=sha(parent), prepared_sha256=sha(prepared), byte_identical=False,
            policy_behavior_unchanged=True, initial_native_behavior_preserved=True,
            comparator_wins=baseline['summary']['wins'], comparator_games=672,
            baseline_unique_games=672, baseline_panel_entries=800, new_baseline_games=0))
        for band in bands:
            with (ROOT/f'runs/training_reservation_{band}.json').open('x') as file:
                json.dump(dict(run=str(out), start=band, stop_exclusive=band+1000000), file)
        write(out/'started.json', dict(time=time.time(), status='prepared_not_launched'))
        write(out/'goal_control.json', dict(status='prepared', goal_status='active', automatic_learning=True,
            publication=False, next_action='One cumulative read-only review, then two fixed-rate CPU PPO repeats.'))
        check_plan(original); check_plan(plan)
        print(json.dumps(dict(out=str(out), prepared=True, rate=rate, state_exact=True,
                              comparator_wins=baseline['summary']['wins'], new_baseline_games=0,
                              runner_change='reporter import only; simulation/training/resume code preserved',
                              frozen_files=len(plan['frozen_sha256']))))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--rate', type=float, default=1e-5)
    args = parser.parse_args(); prepare(args.source.resolve(), args.out.resolve(), args.rate)
