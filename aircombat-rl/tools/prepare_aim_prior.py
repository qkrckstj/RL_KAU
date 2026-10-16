"""Prepare fixed-prior3 only after the unchanged-prior6 study is analyzed."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import time

from experiments.rear_async.storage import identity, read, sha, verify, write
from tools.aim_prior_bias import change_prior, learner_state
from tools.rear_async_cpu import ROOT, check_plan


def select_parent(entries):
    eligible = [(p, e) for p, e in entries
                if e['kind'] == 'development' and e['summary']['games'] == 672]
    if not eligible:
        raise ValueError('No complete672native parent')
    return max(eligible, key=lambda item: (
        item[1]['summary']['wins'],
        item[1]['summary']['damage_dealt']-item[1]['summary']['damage_received']))


def prepare(source, out):
    import psutil
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Use fresh output under runs')
    if not (source/'training_complete.json').exists() or (source/'training_failure.json').exists():
        raise RuntimeError('Source not successfully terminal')
    if read(source/'evaluation_complete.json')['status'] != 'queue_drained' or list((source/'evaluations').glob('*/blocked.json')):
        raise RuntimeError('Source evaluations incomplete or blocked')
    for proc in psutil.process_iter(['cmdline', 'status']):
        args = proc.info['cmdline'] or []
        if (proc.info['status'] != psutil.STATUS_ZOMBIE
                and any(Path(a).name == 'aim_cpu_runner.py' or a == 'tools.rear_async_cpu' for a in args)
                and any(a in args for a in ('launch', 'evaluate'))):
            raise RuntimeError('Existing native trainer/evaluator: '+str(proc.pid))
    if psutil.virtual_memory().available < 10*2**30:
        raise MemoryError('Need10GiB available')
    locks = []
    try:
        for name in ('training.lock', 'evaluation.lock', 'launch.lock'):
            lock = (source/name).open('a'); locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        original = read(source/'plan.json'); check_plan(original)
        review = read(source/'TWO_REPEAT_REVIEW.json'); audit = read(source/'CYCLE_LOG_AUDIT.json')
        if review['status'] != 'two_repeats_complete_analyzed' or set(audit['training']) != {str(s) for s in original['seeds']}:
            raise ValueError('Source repeats not completely analyzed')
        for seed in original['seeds']:
            record = audit['training'][str(seed)]
            if record['all_episodes']['games'] != read(source/f's{seed}_result.json')['training_games'] or record['nonfinite_ppo_fields']:
                raise ValueError('Completed training audit differs or nonfinite')
        if (original['control_guidance_range'] != 12000.
                or original['control_guidance_coefficient'] != .05
                or original['ppo']['ent_coef'] != .003
                or original.get('prior_bias', 6.) != 6.):
            raise ValueError('Expected unchanged12km/prior6control')
        if (original['monitor_ics'] != 4 or original['development_ics'] != 8
                or original['monitor_action_seeds'] != [4900, 4901]
                or original['monitor_band'] != original['development_band']):
            raise ValueError('Expected fixed128subset monitor and672development')
        entries = [(p, read(p)) for p in (source/'evaluations').glob('*/completion.json')]
        completion, best = select_parent(entries)
        checkpoint = Path(best['checkpoint']).resolve(); metadata = verify(checkpoint)
        if best['baseline']:
            if checkpoint != Path(read(source/'baseline_checkpoint.json')['path']).resolve():
                raise ValueError('Selected baseline checkpoint differs')
            input_folder = Path(best['reused_from_evaluation'])/'input'
        else:
            if not checkpoint.is_relative_to(source.resolve()/'checkpoints') or metadata['config_id'] != original['config_id']:
                raise ValueError('Selected trained parent outside source namespace')
            input_folder = completion.parent/'input'
        if (metadata['checkpoint_id'] != best['checkpoint_id']
                or metadata['additional_steps'] != best.get('source_additional_steps', best['additional_steps'])
                or verify(input_folder) != metadata):
            raise ValueError('Evaluated parent identity differs')
        paths = sorted(completion.parent.glob('cells/*.json')); rows = [read(p) for p in paths]
        expected = {(f['id'], original['development_band']+i, seat, sampler)
                    for f in original['development_opponents'] for i in range(8)
                    for seat in ('red', 'blue') for sampler in (4900, 4901)}
        keys = {tuple(r[k] for k in ('opponent', 'seed', 'seat', 'action_seed')) for r in rows}
        if len(rows) != 672 or keys != expected or sum(r['result'] == 'win' for r in rows) != best['summary']['wins']:
            raise ValueError('Parent cells incomplete or unpaired')
        from experiments.aim_async.evaluate import profile
        measured = profile(rows)
        for key in ('games', 'wins', 'damage_dealt', 'damage_received', 'first_shot_advantage_rate'):
            if abs(measured[key]-best['summary'][key]) > 1e-12:
                raise ValueError('Parent profile differs')
        monitor_ids = {f['id'] for f in original['monitor_opponents']}
        monitor_rows = [r for r in rows if r['opponent'] in monitor_ids
                        and r['seed'] < original['development_band']+4]
        if len(monitor_rows) != 128:
            raise ValueError('Monitor subset differs')
        seeds, bands = [46100, 46101], [2600400000, 2610400000]
        for path in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            reservation = read(path)
            if any(b < reservation['stop_exclusive'] and b+1000000 > reservation['start'] for b in bands):
                raise ValueError('Reserved training band: '+str(path))
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1); model = PPO.load(checkpoint/'learner.zip', device='cpu')
        if model.policy.prior_bias != 6. or model.n_envs != 32:
            raise ValueError('Expected prior6parent and32environments')
        for key in ('n_steps', 'n_epochs', 'batch_size', 'gamma', 'gae_lambda', 'target_kl', 'ent_coef', 'max_grad_norm'):
            if getattr(model, key) != original['ppo'][key]:
                raise ValueError('Parent PPO setting differs: '+key)
        if model.lr_schedule(1.) != original['ppo']['learning_rate'] or model.clip_range(1.) != original['ppo']['clip_range']:
            raise ValueError('Parent schedules differ')
        if model.num_timesteps != metadata['total_steps']:
            raise ValueError('Parent counters differ')
        code = (source/'aim_cpu_runner.py').read_text()
        old = 'assert model.policy.prior_bias==6 and model.gamma==.999 and model.gae_lambda==.99'
        if code.count(old) != 1:
            raise ValueError('Runner prior boundary differs')
        code = code.replace(old, "assert model.policy.prior_bias==plan['prior_bias'] and model.gamma==.999 and model.gae_lambda==.99")
        old_report = 'from tools.aim_cpu_continuation_reporting import update'
        if code.count(old_report) != 1:
            raise ValueError('Runner reporting boundary differs')
        code = code.replace(old_report, 'from tools.aim_cpu_prior_reporting import update')
        compile(code, 'aim_cpu_runner.py', 'exec')
        change = change_prior(model, 3.)
        out.mkdir(parents=True); prepared = out/'prior3_parent.zip'; model.save(prepared)
        restored = PPO.load(prepared, device='cpu')
        if (restored.policy.prior_bias != 3. or restored.policy_kwargs['prior_bias'] != 3.
                or learner_state(restored) != change['before']):
            raise ValueError('Saved prior3or actor/critic/Adam/counters differ')
        runner = out/'aim_cpu_runner.py'; runner.write_text(code)
        plan = copy.deepcopy(original)
        plan.update(parent=str(prepared.relative_to(ROOT)), parent_source=str(checkpoint/'learner.zip'),
            selected_parent_checkpoint=str(checkpoint), source_run=str(source),
            seeds=seeds, training_bands=bands, prior_bias=3., comparator_prior_bias=6.,
            intervention=dict(factor='fixed_additive_teacher_prior', control=6., value=3.,
                unchanged='actor/critic tensors,Adam/counters,teacher,43features,rewards,PPO,48foes,quarter0.5,native physics/guns/actions'),
            comparison_scope='Prior3training candidates versus unchanged prior6native parent;cached cells describe comparator only,not an evaluated prior3zero-step model.',
            parent_choice='Best measured complete672native prior6parent including cached control;provisional consumed-development choice',
            baseline_unique_games=672, baseline_panel_entries=800)
        inputs = [prepared, runner, Path(__file__).resolve(), ROOT/'tools/aim_prior_bias.py',
                  ROOT/'tools/aim_cpu_prior_reporting.py', source/'plan.json', source/'CYCLE_LOG_AUDIT.json',
                  source/'TWO_REPEAT_REVIEW.json', source/'PARENT_TURN_MECHANISM_REVIEW.json',
                  source/'TEACHER_BIAS_OFFLINE_REVIEW.json', completion, checkpoint/'READY.json', *paths]
        inputs.extend(checkpoint/name for name in metadata['files'])
        plan['frozen_sha256'].update({str(p.relative_to(ROOT)): sha(p) for p in inputs})
        plan['config_id'] = identity({k:v for k,v in plan.items() if k not in ('config_id', 'frozen_sha256')})
        write(out/'plan.json', plan)
        for kind, cells in [('development', rows), ('monitor', monitor_rows)]:
            folder = out/'evaluations'/identity(dict(config_id=plan['config_id'], parent=metadata['checkpoint_id'], kind=kind))
            entry = dict(best, baseline=True, kind=kind, seed=seeds[0], additional_steps=0,
                source_additional_steps=metadata['additional_steps'], summary=profile(cells), win_gain=None,
                status='reused_unchanged_prior6_comparator', actual_new_games=0,
                new_games_in_final_attempt=0, reused_cells_in_final_attempt=len(cells), wall_seconds=0.,
                checkpoint=str(checkpoint), reused_from_run=str(source), reused_from_evaluation=str(completion.parent),
                per_opponent={foe:profile([r for r in cells if r['opponent']==foe]) for foe in sorted({r['opponent'] for r in cells})},
                reuse_basis='Unchanged prior6parent comparator only;prior3initial learner changes inference and is not represented by these cells.')
            write(folder/'completion.json', entry); write(out/'candidates'/f'{folder.name}.json', entry)
            for row in cells:
                write(folder/'cells'/f"{identity({k:row[k] for k in ('opponent','seed','seat','action_seed')})}.json", row)
        write(out/'baseline_checkpoint.json', dict(path=str(checkpoint)))
        write(out/'parent_verification.json', dict(**change, restored=learner_state(restored),
            source_checkpoint=str(checkpoint), source_sha256=sha(checkpoint/'learner.zip'),
            prepared_sha256=sha(prepared), byte_identical=False, policy_behavior_unchanged=False,
            comparator_wins=measured['wins'], comparator_games=672, new_baseline_games=0,
            baseline_unique_games=672, baseline_panel_entries=800, prior3_initial_evaluated=False))
        for band in bands:
            with (ROOT/f'runs/training_reservation_{band}.json').open('x') as f:
                json.dump(dict(run=str(out), start=band, stop_exclusive=band+1000000), f)
        write(out/'started.json', dict(time=time.time(), status='prepared_not_launched'))
        write(out/'goal_control.json', dict(status='prepared', goal_status='active', automatic_learning=True,
            publication=False, effective_objective=plan['objective'],
            next_action='One read-only review then fixed-prior3native PPO study'))
        check_plan(original); check_plan(plan)
        print(json.dumps(dict(out=str(out), prepared=True, tensor_Adam_counters_exact=True,
                              source_prior=6., target_prior=3., comparator_wins=measured['wins'], new_baseline_games=0)))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); prepare(args.source.resolve(), args.out.resolve())
