"""Align a fixed training reference with the preserved classroom benchmark parent."""
import argparse
import copy
import fcntl
import math
from pathlib import Path
import time

from experiments.rear_async.storage import identity, read, sha, verify, write
from tools.aim_prior_bias import learner_state
from experiments.aim_async.reference_archive import rebind_reference_archive
from tools.rear_async_cpu import ROOT, check_plan


def terminal(folder):
    if not (folder / 'training_complete.json').exists() or (folder / 'training_failure.json').exists():
        raise ValueError('Training not successfully terminal')
    if read(folder / 'evaluation_complete.json')['status'] != 'queue_drained':
        raise ValueError('Evaluations not drained')
    for file in (folder / 'queue').glob('*.json'):
        job = read(file); result = folder / 'evaluations' / job['job_id']
        if not (result / 'completion.json').exists() or (result / 'blocked.json').exists():
            raise ValueError('Unfinished or blocked evaluation')


def prepare(source, evidence, out):
    import psutil
    import torch
    from experiments.aim_async.reference_ppo import ReferenceAnchoredPPO
    from experiments.aim_async.evaluate import profile
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Use a fresh run under runs')
    for proc in psutil.process_iter(['cmdline', 'status']):
        args = proc.info['cmdline'] or []
        if proc.info['status'] != psutil.STATUS_ZOMBIE and any(
                Path(a).name == 'aim_cpu_runner.py' or a == 'tools.rear_async_cpu' for a in args
        ) and any(a in args for a in ('launch', 'evaluate')):
            raise RuntimeError('Native trainer/evaluator already active')
    if psutil.virtual_memory().available < 10 * 2**30:
        raise MemoryError('Need10GiB free')
    locks = []
    try:
        for folder in dict.fromkeys((source, evidence)):
            for name in ('training.lock', 'evaluation.lock', 'launch.lock'):
                lock = (folder / name).open('a'); locks.append(lock)
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            terminal(folder)
        original = read(source / 'plan.json'); check_plan(original)
        evidence_plan = read(evidence / 'plan.json'); check_plan(evidence_plan)
        audit = read(evidence / 'CYCLE_LOG_AUDIT.json')
        if set(audit['training']) != {str(s) for s in evidence_plan['seeds']}:
            raise ValueError('Both completed repetitions must be audited')
        if any(t['nonfinite_ppo_fields'] for t in audit['training'].values()):
            raise ValueError('Nonfinite logged PPO state')
        if original['aim_coefficients'] != dict(potential=.05, damage=.5, received=.75):
            raise ValueError('Expected stable reward control')
        if not original['actor_features_trainable'] or original['qualified_parent_full_wins'] != 510:
            raise ValueError('Expected preserved510 feature-learning source')
        if original['ppo']['learning_rate'] != 1e-6 or original['ppo']['ent_coef'] != .003:
            raise ValueError('Expected fixed1e-6/entropy0.003 control')
        checkpoint = Path(read(source / 'baseline_checkpoint.json')['path']).resolve()
        meta = verify(checkpoint); parent = ROOT / original['parent']
        if sha(parent) != sha(checkpoint / 'learner.zip'):
            raise ValueError('Starting learner differs from qualified comparator')
        panels = []
        for kind in ('monitor', 'development'):
            paths = [p for p in (source / 'evaluations').glob('*/completion.json')
                     if read(p)['baseline'] and read(p)['kind'] == kind]
            if len(paths) != 1:
                raise ValueError('Ambiguous baseline')
            file = paths[0]; entry = read(file)
            files = sorted((file.parent / 'cells').glob('*.json')); rows = [read(p) for p in files]
            samplers = original['monitor_action_seeds'] if kind == 'monitor' else [4900, 4901]
            expected = {(foe['id'], original[kind + '_band'] + i, seat, action)
                        for foe in original[kind + '_opponents'] for i in range(original[kind + '_ics'])
                        for seat in ('red', 'blue') for action in samplers}
            if entry['checkpoint_id'] != meta['checkpoint_id'] or len(rows) != len(expected) or {
                    tuple(r[k] for k in ('opponent', 'seed', 'seat', 'action_seed')) for r in rows} != expected:
                raise ValueError('Cached baseline identity/conditions differ')
            measured = profile(rows)
            if any(measured[k] != v and not (v is not None and measured[k] is not None and
                    math.isclose(v, measured[k], abs_tol=1e-12, rel_tol=0)) for k, v in entry['summary'].items()):
                raise ValueError('Cached summary differs')
            panels.append((file, entry, files, rows))
        if panels[1][1]['summary']['wins'] != 510 or len(panels[1][3]) != 672:
            raise ValueError('Expected510/672 comparator')
        seeds, bands = [47202, 47203], [3260400000, 3270400000]
        for file in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            r = read(file)
            if any(b < r['stop_exclusive'] and b + 1000000 > r['start'] for b in bands):
                raise ValueError('Training band already reserved')
        torch.set_num_threads(1)
        model = ReferenceAnchoredPPO.load(parent, device='cpu')
        before = learner_state(model)
        if model.ent_coef != .003 or model.reference_policy_hash != original['reference_policy_sha256']:
            raise ValueError('Starting entropy/reference differ from stable control')
        out.mkdir(parents=True); prepared = out / 'reference510_parent.zip'
        migration = rebind_reference_archive(parent, prepared, original['reference_policy_sha256'])
        restored = ReferenceAnchoredPPO.load(prepared, device='cpu')
        if learner_state(restored) != before or restored.ent_coef != .003:
            raise ValueError('Prepared learner roundtrip differs')
        restored.attach_reference(prepared, original['reference_kl_coefficient'])
        if restored.reference_kl(torch.zeros(4, 43)).item() != 0.:
            raise ValueError('Initial self-reference KL is not zero')
        if any(p.requires_grad for p in restored.reference_policy.parameters()):
            raise ValueError('Reference actor must remain frozen')
        proof = dict(before=before, after=learner_state(restored), state_exact=True,
                     **migration, initial_reference_kl_zero=True)
        runner = out / 'aim_cpu_runner.py'
        code = (source / 'aim_cpu_runner.py').read_text()
        old = 'from tools.aim_cpu_selected510_reporting import update'
        if code.count(old) != 1:
            raise ValueError('Reporter boundary differs')
        code = code.replace(old, 'from tools.aim_cpu_reference510_reporting_v2 import update')
        compile(code, str(runner), 'exec'); runner.write_text(code)
        plan = copy.deepcopy(original)
        plan['scope'] = 'Full actor-feature PPO from selected510; fixed KL reference old500→same510, coefficient0.1; native environment unchanged.'
        plan.update(previous_reference_archive=original['reference_archive'],
                    previous_reference_policy_sha256=original['reference_policy_sha256'],
                    reference_archive=str(prepared.relative_to(ROOT)),
                    reference_policy_sha256=migration['new_reference_policy_sha256'])
        plan.update(parent=str(prepared.relative_to(ROOT)), parent_source=str(parent.relative_to(ROOT)),
                    source_run=str(source), evidence_run=str(evidence), seeds=seeds, training_bands=bands,
                    intervention=dict(factor='fixed_reference_actor',
                        control_policy_sha256=original['reference_policy_sha256'],
                        value_policy_sha256=migration['new_reference_policy_sha256'],
                        comparison_basis='Stable510 control; actor/Adam/counters/rewards/entropy fixed',
                        actor_features_trainable=True,
                        unchanged='Policy/Adam/counters/rewards/entropy/PPO/inputs/teacher/prior/native environment/rosters/evaluation/budget'),
                    comparison_scope='Adaptive development comparison, fresh training RNGs; not unseen final evidence')
        inputs = [prepared, runner, Path(__file__), ROOT / 'tools/aim_cpu_reference510_reporting_v2.py',
                  ROOT / 'tests/test_aim_reference_rebind.py', ROOT / 'experiments/aim_async/reference_archive.py', source / 'plan.json', source / 'TWO_REPEAT_REVIEW.json',
                  evidence / 'CYCLE_LOG_AUDIT.json', evidence / 'TWO_REPEAT_REVIEW.json', evidence / 'NEXT_REFERENCE510_PLAN.md']
        for file, _, files, _ in panels:
            inputs.extend([file, *files])
        plan['frozen_sha256'].update({str(p.resolve().relative_to(ROOT)): sha(p) for p in inputs})
        plan['config_id'] = identity({k: v for k, v in plan.items() if k not in ('config_id', 'frozen_sha256')})
        write(out / 'plan.json', plan)
        for file, entry, _, rows in panels:
            folder = out / 'evaluations' / identity(dict(config_id=plan['config_id'], kind=entry['kind']))
            reused = dict(entry, seed=seeds[0], wall_seconds=0., actual_new_games=0,
                          new_games_in_final_attempt=0, reused_cells_in_final_attempt=len(rows), reused_from_sibling_games=0,
                          reused_from_run=str(source), reused_from_evaluation=str(file.parent),
                          reuse_basis='Identical initial policy/export; only frozen reference hash/path changes; same evaluation conditions')
            write(folder / 'completion.json', reused); write(out / 'candidates' / f'{folder.name}.json', reused)
            for row in rows:
                write(folder / 'cells' / f"{identity({k: row[k] for k in ('opponent','seed','seat','action_seed')})}.json", row)
        write(out / 'baseline_checkpoint.json', dict(path=str(checkpoint)))
        write(out / 'parent_verification.json', dict(proof, restored=learner_state(restored), byte_identical=False,
              initial_native_behavior_preserved=True, comparator_wins=510, comparator_games=672,
              baseline_unique_games=672, baseline_panel_entries=800, new_baseline_games=0))
        for band in bands:
            path = ROOT / f'runs/training_reservation_{band}.json'
            with path.open('x') as f:
                import json
                json.dump(dict(run=str(out), start=band, stop_exclusive=band + 1000000), f)
        write(out / 'started.json', dict(time=time.time(), status='prepared_not_launched'))
        write(out / 'goal_control.json', dict(status='prepared', goal_status='active', publication=False))
        check_plan(original); check_plan(plan)
        print(dict(prepared=True, old_reference=original['reference_policy_sha256'], new_reference=plan['reference_policy_sha256'], seeds=seeds, policy_Adam_counters_exact=True, new_baseline_games=0))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('source', 'evidence', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); prepare(a.source.resolve(), a.evidence.resolve(), a.out.resolve())
