"""Strengthen terminal draw utility with the unchanged qualified initial learner."""
import argparse
import copy
import fcntl
import json
import math
import shutil
import time
import zipfile
from pathlib import Path

from experiments.rear_async.storage import identity, read, sha, verify, write
from tools.rear_async_cpu import ROOT, check_plan


def prepare(source, out, penalty=.5):
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a fresh output under runs')
    if not math.isfinite(penalty) or not .25 < penalty <= 1.:
        raise ValueError('Use a stronger finite draw penalty above0.25andatmost1')
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
                or original['ppo']['learning_rate'] != 1e-5
                or original['terminal_draw_penalty'] != .25
                or original['learner_feature_dim'] != 43 or original['prior_bias'] != 6.
                or original['processes'] != 8 or original['workers'] != 32):
            raise ValueError('Expected completed predictive-enemy rate1e-5/draw0.25 control')
        entries = [(p, read(p)) for p in (source/'evaluations').glob('*/completion.json')]
        baseline = next(e for _, e in entries if e['baseline'] and e['kind'] == 'development')
        endpoints = [e for _, e in entries if not e['baseline'] and e['kind'] == 'development']
        if len(endpoints) != 2 or any(e['summary']['wins'] >= baseline['summary']['wins'] for e in endpoints):
            raise ValueError('Review improved/incomplete endpoints before changing draw utility')
        checkpoint = Path(read(source/'baseline_checkpoint.json')['path']).resolve()
        metadata = verify(checkpoint)
        source_evidence = read(source/'parent_verification.json')
        parent = ROOT/original['parent']
        if (not source_evidence['state_exact'] or sha(parent) != source_evidence['prepared_sha256']
                or source_evidence['before'] != source_evidence['restored']):
            raise ValueError('Inherited qualified parent learner differs')
        with zipfile.ZipFile(parent) as current, zipfile.ZipFile(checkpoint/'learner.zip') as qualified:
            for name in ('policy.pth', 'policy.optimizer.pth', 'pytorch_variables.pth'):
                if current.read(name) != qualified.read(name):
                    raise ValueError('Qualified initial learner state differs: '+name)
            serialized = json.loads(current.read('data'))
        for key in ('learning_rate', 'n_steps', 'n_epochs', 'batch_size', 'gamma', 'gae_lambda',
                    'target_kl', 'ent_coef', 'max_grad_norm'):
            if serialized[key] != original['ppo'][key]:
                raise ValueError('Saved source PPO configuration differs: '+key)
        if (serialized['num_timesteps'] != metadata['total_steps'] or serialized['n_envs'] != 32
                or serialized['_n_updates'] != source_evidence['before']['updates']):
            raise ValueError('Saved source counters or environments differ')
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
        seeds, bands = [46900, 46901], [2800400000, 2810400000]
        for path in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            reservation = read(path)
            if any(b < reservation['stop_exclusive'] and b+1000000 > reservation['start'] for b in bands):
                raise ValueError('Reserved training band: '+str(path))
        before = source_evidence['before']
        out.mkdir(parents=True); prepared = out/'draw_strength_parent.zip'
        shutil.copyfile(parent, prepared)
        if sha(prepared) != sha(parent):
            raise ValueError('Copied initial learner bytes differ')
        runner = out/'aim_cpu_runner.py'
        runner_code = (source/'aim_cpu_runner.py').read_text()
        old_report = 'from tools.aim_cpu_learning_rate_reporting import update'
        if runner_code.count(old_report) != 1:
            raise ValueError('Reporter import boundary differs')
        runner_code = runner_code.replace(old_report, 'from tools.aim_cpu_draw_strength_reporting import update')
        compile(runner_code, str(runner), 'exec'); runner.write_text(runner_code)
        plan = copy.deepcopy(original); plan['terminal_draw_penalty'] = penalty
        plan.update(parent=str(prepared.relative_to(ROOT)), parent_source=str(parent), source_run=str(source),
                    seeds=seeds, training_bands=bands, scope='Fixed stronger draw utility from qualified499reference, rate1e-5 unchanged.',
                    intervention=dict(factor='terminal_draw_utility', control=-.25, value=-penalty,
                        unchanged='initiallearnerbytes/NN/initializedAdam/counters/43features/predictive reward/ownaim/HP/control guidance/prior6/teacher/PPO rate1e-5/48foes/quarter0.5/nativephysics/guns/verdict/evaluation'),
                    comparison_scope='Adaptive single-scalar terminaldraw reward study; initial learner bytes identical to control; native672development,not unseen final.',
                    parent_choice='Preserved strongest completed672reference; provisional development choice')
        inputs = [prepared, runner, Path(__file__).resolve(), ROOT/'tests/test_aim_draw_strength.py',
                  source/'plan.json', source/'parent_verification.json', source/'CYCLE_LOG_AUDIT.json',
                  source/'TWO_REPEAT_REVIEW.json', source/'S46810_COMPLETED_REVIEW.json',
                  source/'S46811_COMPLETED_REVIEW.json', source/'REPEAT_RESULTS.md',
                  source/'DRAW_STRENGTH_NEXT_STUDY.md', source/'PAIRED_MUTUAL_DRAW_DIAGNOSIS.json',
                  source/'CONDITIONAL_RATE_POLICY_DRIFT.json', source/'REVIEW.json',
                  ROOT/'tools/aim_cpu_draw_strength_reporting.py',
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
                status='reused_byte_identical_initial_learner_stronger_draw', actual_new_games=0,
                new_games_in_final_attempt=0, reused_cells_in_final_attempt=len(rows), wall_seconds=0.,
                reused_from_run=str(source), reused_from_evaluation=str(p.parent),
                reuse_basis='Control initiallearner ZIP byte-identical; qualifiedoriginalNN/Adam/counters/43features/prior unchanged. Only terminaldraw learning utility changes; native initial actor identical.',
                per_opponent={foe:profile([r for r in rows if r['opponent'] == foe]) for foe in sorted({r['opponent'] for r in rows})})
            write(folder/'completion.json', reused); write(out/'candidates'/f'{folder.name}.json', reused)
            for row in rows:
                write(folder/'cells'/f"{identity({k:row[k] for k in ('opponent', 'seed', 'seat', 'action_seed')})}.json", row)
        write(out/'baseline_checkpoint.json', dict(path=str(checkpoint)))
        write(out/'parent_verification.json', dict(before=before, after=before, restored=before, state_exact=True,
            source_sha256=sha(parent), prepared_sha256=sha(prepared), byte_identical=True,
            byte_identical_to_control=True, state_entries_identical_to_qualified_original=True,
            qualified_original_sha256=sha(checkpoint/'learner.zip'), fixed_learning_rate=1e-5,
            source_draw_penalty=.25, target_draw_penalty=penalty,
            policy_behavior_unchanged=True, initial_native_behavior_preserved=True,
            comparator_wins=baseline['summary']['wins'], comparator_games=672,
            baseline_unique_games=672, baseline_panel_entries=800, new_baseline_games=0))
        for band in bands:
            with (ROOT/f'runs/training_reservation_{band}.json').open('x') as file:
                json.dump(dict(run=str(out), start=band, stop_exclusive=band+1000000), file)
        write(out/'started.json', dict(time=time.time(), status='prepared_not_launched'))
        write(out/'goal_control.json', dict(status='prepared', goal_status='active', automatic_learning=True,
            publication=False, next_action='One cumulative read-only review, then two fixed stronger-draw CPU PPO repeats.'))
        check_plan(original); check_plan(plan)
        print(json.dumps(dict(out=str(out), prepared=True, draw_penalty=penalty, state_exact=True, byte_identical=True,
                              comparator_wins=baseline['summary']['wins'], new_baseline_games=0,
                              runner_change='reporter import only; simulation/training/resume code preserved',
                              frozen_files=len(plan['frozen_sha256']))))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--draw-penalty', type=float, default=.5)
    args = parser.parse_args(); prepare(args.source.resolve(), args.out.resolve(), args.draw_penalty)
