"""Prepare a learning-only public heading-rate feedback study after gated-prior regression."""
import argparse
import copy
import fcntl
import json
import shutil
from pathlib import Path
import time

from experiments.rear_async.storage import identity, read, sha, verify, write
from tools.aim_prior_bias import learner_state
from experiments.aim_async.policy import AimPriorPolicy
from tools.rear_async_cpu import ROOT, check_plan


def select_parent(entries):
    eligible = [(p, e) for p, e in entries
                if e['kind'] == 'development' and e['summary']['games'] == 672]
    if not eligible:
        raise ValueError('No complete672native parent')
    return max(eligible, key=lambda item: (
        item[1]['summary']['wins'],
        item[1]['summary']['damage_dealt']-item[1]['summary']['damage_received']))


def comparator_origin(completion, selected):
    folder=Path(completion).resolve().parent;visited=set()
    while True:
        if folder in visited:raise ValueError('Comparator provenance cycle')
        visited.add(folder);entry=read(folder/'completion.json')
        if (entry['checkpoint_id']!=selected['checkpoint_id'] or entry['kind']!='development'
                or entry['summary']['games']!=672
                or Path(entry['checkpoint']).resolve()!=Path(selected['checkpoint']).resolve()):
            raise ValueError('Comparator provenance identity differs')
        if (folder/'input').is_dir():return folder
        if not entry['baseline']:raise ValueError('Unverified comparator origin')
        folder=Path(entry['reused_from_evaluation']).resolve()


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
                or original.get('prior_bias') != 6. or original.get('conflict_prior_bias') != 3.
                or original.get('comparator_prior_bias') != 6.):
            raise ValueError('Expected completed12km/state-prior6conflict3control')
        if (original['monitor_ics'] != 4 or original['development_ics'] != 8
                or original['monitor_action_seeds'] != [4900, 4901]
                or original['monitor_band'] != original['development_band']):
            raise ValueError('Expected fixed128subset monitor and672development')
        entries = [(p, read(p)) for p in (source/'evaluations').glob('*/completion.json')]
        reference=next(e for _,e in entries if e['baseline'] and e['kind']=='development')
        endpoints=[e for _,e in entries if not e['baseline'] and e['kind']=='development']
        if len(endpoints)!=2 or any(e['summary']['games']!=672 or e['summary']['wins']>=reference['summary']['wins'] for e in endpoints):
            raise ValueError('State-prior did not regress in both endpoints;reconsider next factor')
        diagnostic_path=ROOT/'runs/aim_cpu_prior3_20261013/HEADING_RATE_DIAGNOSIS.json'
        diagnostic=read(diagnostic_path)
        trace_path=Path(diagnostic['source'])
        if (diagnostic['source_policy_prior']!=6. or sha(trace_path)!=diagnostic['source_sha256']
                or not read(trace_path.parent/'summary.json')['exact_cached_reproduction']):
            raise ValueError('Stored public heading-rate diagnostic differs')
        completion, best = select_parent(entries)
        if not best['baseline']:raise ValueError('Expected unchanged prior6reference as strongest parent')
        checkpoint = Path(best['checkpoint']).resolve(); metadata = verify(checkpoint)
        if best['baseline']:
            if checkpoint != Path(read(source/'baseline_checkpoint.json')['path']).resolve():
                raise ValueError('Selected baseline checkpoint differs')
            origin=comparator_origin(completion,best)
            input_folder=origin/'input'
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
        origin_rows=[read(p) for p in sorted(origin.glob('cells/*.json'))]
        indexed=lambda values:{tuple(r[k] for k in ('opponent','seed','seat','action_seed')):r for r in values}
        if len(origin_rows)!=672 or indexed(origin_rows)!=indexed(rows):
            raise ValueError('Cached comparator differs from original native cells')
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
        seeds, bands = [46300, 46301], [2640400000, 2650400000]
        for path in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            reservation = read(path)
            if any(b < reservation['stop_exclusive'] and b+1000000 > reservation['start'] for b in bands):
                raise ValueError('Reserved training band: '+str(path))
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1); model = PPO.load(checkpoint/'learner.zip', device='cpu')
        if type(model.policy) is not AimPriorPolicy or model.policy.prior_bias != 6. or model.n_envs != 32:
            raise ValueError('Expected prior6parent and32environments')
        for key in ('n_steps', 'n_epochs', 'batch_size', 'gamma', 'gae_lambda', 'target_kl', 'ent_coef', 'max_grad_norm'):
            if getattr(model, key) != original['ppo'][key]:
                raise ValueError('Parent PPO setting differs: '+key)
        if model.lr_schedule(1.) != original['ppo']['learning_rate'] or model.clip_range(1.) != original['ppo']['clip_range']:
            raise ValueError('Parent schedules differ')
        if model.num_timesteps != metadata['total_steps']:
            raise ValueError('Parent counters differ')
        code = (source/'aim_cpu_runner.py').read_text()
        substitutions = [
            ('from experiments.aim_async.guidance_range_worker import make_worker',
             'from experiments.aim_async.heading_rate_worker import make_worker'),
            ('from experiments.aim_async.state_prior_policy import StatePriorPolicy',
             'from experiments.aim_async.policy import AimPriorPolicy'),
            ("assert isinstance(model.policy,StatePriorPolicy) and model.policy.prior_bias==plan['prior_bias'] and model.policy.conflict_prior_bias==plan['conflict_prior_bias'] and model.gamma==.999 and model.gae_lambda==.99",
             "assert type(model.policy) is AimPriorPolicy and model.policy.prior_bias==plan['prior_bias'] and model.gamma==.999 and model.gae_lambda==.99"),
            ('from tools.aim_cpu_state_prior_export import export',
             'from tools.aim_cpu_export import export'),
            ('from tools.aim_cpu_state_prior_reporting import update',
             'from tools.aim_cpu_heading_rate_reporting import update')]
        for old,new in substitutions:
            if code.count(old)!=1:raise ValueError('Runner boundary differs: '+old)
            code=code.replace(old,new)
        compile(code,'aim_cpu_runner.py','exec')
        before=learner_state(model)
        change=dict(before=before,after=before,state_exact=True,policy_behavior_unchanged=True)
        out.mkdir(parents=True);prepared=out/'heading_rate_parent.zip'
        shutil.copyfile(checkpoint/'learner.zip',prepared)
        restored=PPO.load(prepared,device='cpu')
        if (type(restored.policy) is not AimPriorPolicy or restored.policy.prior_bias!=6.
                or restored.policy_kwargs.get('prior_bias',6.)!=6.
                or learner_state(restored)!=before
                or sha(prepared)!=sha(checkpoint/'learner.zip')):
            raise ValueError('Copied initial learner bytes/policy/Adam/counters differ')
        runner = out/'aim_cpu_runner.py'; runner.write_text(code)
        plan = copy.deepcopy(original)
        plan.pop('conflict_prior_bias',None);plan.pop('prior_rule',None)
        plan.update(parent=str(prepared.relative_to(ROOT)), parent_source=str(checkpoint/'learner.zip'),
            selected_parent_checkpoint=str(checkpoint),source_run=str(source),
            seeds=seeds,training_bands=bands,prior_bias=6.,comparator_prior_bias=6.,
            heading_feedback='public_euler_heading_rate_v1',
            intervention=dict(factor='control_guidance_heading_rate',control='bodyr',value='publicEulerHeadingRate',
                unchanged='initial learner bytes/actor/critic/Adam/counters/43features/teacher/prior6/PPO/48foes/quarter0.5/all other rewards/native physics/guns/actions'),
            comparison_scope='Only learning potential uses public Euler heading feedback;byte-identical initial prior6native inference uses cached672paired baseline cells.',
            parent_choice='Best complete672native prior6parent including unchanged comparator;provisional consumed-development choice',
            baseline_unique_games=672,baseline_panel_entries=800)
        inputs=[prepared,runner,Path(__file__).resolve(),ROOT/'tools/aim_prior_bias.py',
                ROOT/'tools/aim_cpu_heading_rate_reporting.py',ROOT/'experiments/aim_async/heading_rate_worker.py',
                ROOT/'tests/test_aim_heading_rate.py',ROOT/'tests/test_prepare_aim_heading_rate.py',
                source/'plan.json',source/'CYCLE_LOG_AUDIT.json',source/'TWO_REPEAT_REVIEW.json',
                source/'S46200_COMPLETED_REVIEW.json',source/'S46201_COMPLETED_REVIEW.json',
                source/'S46200_OUTCOME_MECHANISM_REVIEW.json',diagnostic_path,trace_path,
                trace_path.parent/'summary.json',origin/'completion.json',completion,checkpoint/'READY.json',*paths]
        inputs.extend(checkpoint/name for name in metadata['files'])
        plan['frozen_sha256'].update({str(p.relative_to(ROOT)): sha(p) for p in inputs})
        plan['config_id'] = identity({k:v for k,v in plan.items() if k not in ('config_id', 'frozen_sha256')})
        write(out/'plan.json', plan)
        for kind, cells in [('development', rows), ('monitor', monitor_rows)]:
            folder = out/'evaluations'/identity(dict(config_id=plan['config_id'], parent=metadata['checkpoint_id'], kind=kind))
            entry = dict(best, baseline=True, kind=kind, seed=seeds[0], additional_steps=0,
                source_additional_steps=metadata['additional_steps'], summary=profile(cells), win_gain=None,
                status='reused_byte_identical_initial_prior6', actual_new_games=0,
                new_games_in_final_attempt=0, reused_cells_in_final_attempt=len(cells), wall_seconds=0.,
                checkpoint=str(checkpoint), reused_from_run=str(source), reused_from_evaluation=str(completion.parent),
                per_opponent={foe:profile([r for r in cells if r['opponent']==foe]) for foe in sorted({r['opponent'] for r in cells})},
                reuse_basis='Byte-identical initial prior6learner including actor/critic/Adam/counters. Learning-only heading potential does not change native initial inference;cached paired cells remain valid.')
            write(folder/'completion.json', entry); write(out/'candidates'/f'{folder.name}.json', entry)
            for row in cells:
                write(folder/'cells'/f"{identity({k:row[k] for k in ('opponent','seed','seat','action_seed')})}.json", row)
        write(out/'baseline_checkpoint.json', dict(path=str(checkpoint)))
        write(out/'parent_verification.json', dict(**change, restored=learner_state(restored),
            source_checkpoint=str(checkpoint), source_sha256=sha(checkpoint/'learner.zip'),
            prepared_sha256=sha(prepared), byte_identical=True,
            comparator_wins=measured['wins'], comparator_games=672, new_baseline_games=0,
            baseline_unique_games=672, baseline_panel_entries=800, initial_native_behavior_preserved=True))
        for band in bands:
            with (ROOT/f'runs/training_reservation_{band}.json').open('x') as f:
                json.dump(dict(run=str(out), start=band, stop_exclusive=band+1000000), f)
        write(out/'started.json', dict(time=time.time(), status='prepared_not_launched'))
        write(out/'goal_control.json', dict(status='prepared', goal_status='active', automatic_learning=True,
            publication=False, effective_objective=plan['objective'],
            next_action='One cumulative read-only code/artifact review then two public-heading-rate nativeCPU PPO repeats'))
        check_plan(original); check_plan(plan)
        print(json.dumps(dict(out=str(out), prepared=True, tensor_Adam_counters_exact=True,
                              base_prior=6., heading_feedback=plan['heading_feedback'], comparator_wins=measured['wins'], new_baseline_games=0)))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); prepare(args.source.resolve(), args.out.resolve())
