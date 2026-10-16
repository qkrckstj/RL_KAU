"""Prepare a public motion input study after two completed43feature controls."""
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
from experiments.aim_async.rate_input_policy import RateInputPriorPolicy, expand_learner, projected_learner_state, FIRST
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


def verify_initial_behavior(source,restored,out,teacher,trace_path):
    import numpy as np
    import torch
    from experiments.aim_async.features import features as old_features
    from experiments.aim_async.rate_input_features import features as new_features
    from tools.aim_cpu_export import export as old_export
    from tools.aim_cpu_rate_input_export import export as new_export
    from tools.league_residual_action_mode import build_variant
    from tools.policies import load
    records=[json.loads(line) for line in trace_path.read_text().splitlines() if line]
    if len(records)!=1609:raise ValueError('Expected original consumed1609frame trace')
    old=np.stack([old_features(r['pre_raw_obs'],r['teacher_action']) for r in records])
    new=np.stack([new_features(r['pre_raw_obs'],r['teacher_action']) for r in records])
    with torch.no_grad():
        a=source.policy.get_distribution(torch.from_numpy(old)).distribution.logits
        b=restored.policy.get_distribution(torch.from_numpy(new)).distribution.logits
        value_a=source.policy.predict_values(torch.from_numpy(old))
        value_b=restored.policy.predict_values(torch.from_numpy(new))
    if not torch.equal(a,b) or not torch.equal(value_a,value_b):
        raise ValueError('Initial actor/critic differs on consumed trace')
    parameters=list(restored.policy.named_parameters())
    optimizer=restored.policy.optimizer.state_dict()
    ids=[i for g in optimizer['param_groups'] for i in g['params']]
    for index,(name,value) in zip(ids,parameters):
        if name not in FIRST:continue
        if torch.count_nonzero(value[:,34]):raise ValueError('New weight column not zero')
        for item in optimizer['state'].get(index,{}).values():
            if isinstance(item,torch.Tensor) and item.ndim==2 and torch.count_nonzero(item[:,34]):
                raise ValueError('New Adam column not zero')
    before=old_export(out/'initial_native/old_greedy',teacher,source.policy,'reference43')
    after=new_export(out/'initial_native/new_greedy',teacher,restored.policy,'initial44')
    specs={'greedy':(before,after)}
    for sampler in (4900,4901):
        specs[str(sampler)]=tuple(build_variant(spec,out/f'initial_native/{label}_{sampler}',sampler,label+str(sampler))
            for label,spec in [('old',before),('new',after)])
    checked={}
    for mode,(old_spec,new_spec) in specs.items():
        old_act=load(ROOT/old_spec['design'],ROOT/old_spec['weights'])[0]
        new_act=load(ROOT/new_spec['design'],ROOT/new_spec['weights'])[0]
        differences=sum(old_act(np.array(row['pre_raw_obs'],dtype=np.float32))!=new_act(np.array(row['pre_raw_obs'],dtype=np.float32)) for row in records)
        if differences:raise ValueError('Native initial actions differ: '+mode)
        checked[mode]=dict(frames=len(records),action_differences=differences)
    return dict(trace=str(trace_path),trace_sha256=sha(trace_path),frames=len(records),
        torch_logits_exact=True,torch_values_exact=True,new_columns_and_moments_zero=True,
        native_modes=checked,new_games=0,
        scope='Projected state and initial inference compatibility on one consumed trace plus zero-column construction;not a new tournament or evidence of better learned behavior.')


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
                or original.get('prior_bias') != 6.
                or original.get('comparator_prior_bias') != 6.
                or original.get('heading_feedback') != 'public_euler_heading_rate_v1'):
            raise ValueError('Expected completed12km/public-heading43control')
        if (original['monitor_ics'] != 4 or original['development_ics'] != 8
                or original['monitor_action_seeds'] != [4900, 4901]
                or original['monitor_band'] != original['development_band']):
            raise ValueError('Expected fixed128subset monitor and672development')
        entries = [(p, read(p)) for p in (source/'evaluations').glob('*/completion.json')]
        reference=next(e for _,e in entries if e['baseline'] and e['kind']=='development')
        endpoints=[e for _,e in entries if not e['baseline'] and e['kind']=='development']
        if len(endpoints)!=2 or any(e['summary']['games']!=672 or e['summary']['wins']>=reference['summary']['wins'] for e in endpoints):
            raise ValueError('Both43feature endpoints must fall below reference;reconsider next factor')
        diagnostic_path=source/'OBSERVABILITY_REVIEW.json'
        diagnostic=read(diagnostic_path);trace_path=Path(diagnostic['source_trace'])
        if (sha(trace_path)!=diagnostic['source_sha256']
                or not diagnostic['derived43_features_identical']
                or not diagnostic['reference43_numpy_logits_identical']
                or not read(trace_path.parent/'summary.json')['exact_cached_reproduction']):
            raise ValueError('Stored public motion observability evidence differs')
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
        seeds, bands = [46400, 46401], [2660400000, 2670400000]
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
            ('from experiments.aim_async.heading_rate_worker import make_worker',
             'from experiments.aim_async.rate_input_worker import make_worker'),
            ('from experiments.aim_async.policy import AimPriorPolicy',
             'from experiments.aim_async.rate_input_policy import RateInputPriorPolicy'),
            ("assert type(model.policy) is AimPriorPolicy and model.policy.prior_bias==plan['prior_bias'] and model.gamma==.999 and model.gae_lambda==.99",
             "assert type(model.policy) is RateInputPriorPolicy and model.observation_space.shape==(44,) and model.policy.prior_bias==plan['prior_bias'] and model.gamma==.999 and model.gae_lambda==.99"),
            ('from tools.aim_cpu_export import export',
             'from tools.aim_cpu_rate_input_export import export'),
            ('from tools.aim_cpu_heading_rate_reporting import update',
             'from tools.aim_cpu_rate_input_reporting import update')]
        for old,new in substitutions:
            if code.count(old)!=1:raise ValueError('Runner boundary differs: '+old)
            code=code.replace(old,new)
        compile(code,'aim_cpu_runner.py','exec')
        before=learner_state(model)
        expanded=expand_learner(model)
        if projected_learner_state(expanded)!=before:raise ValueError('Initial projected learner differs')
        out.mkdir(parents=True);prepared=out/'rate_input_parent.zip';expanded.save(prepared)
        restored=PPO.load(prepared,device='cpu')
        if (type(restored.policy) is not RateInputPriorPolicy or restored.policy.prior_bias!=6.
                or restored.policy_kwargs.get('prior_bias',6.)!=6.
                or restored.observation_space.shape!=(44,)
                or projected_learner_state(restored)!=before):
            raise ValueError('Reloaded projected actor/critic/Adam/counters differ')
        initial=verify_initial_behavior(model,restored,out,original['teacher'],trace_path)
        if learner_state(model)!=before:raise ValueError('Source learner changed during verification')
        change=dict(before=before,after=learner_state(restored),
            projected_restored=projected_learner_state(restored),projected_state_exact=True,
            new_columns_and_moments_zero=initial['new_columns_and_moments_zero'],
            initial_inference=initial,policy_behavior_unchanged_on_trace=True)
        runner = out/'aim_cpu_runner.py'; runner.write_text(code)
        plan = copy.deepcopy(original)
        plan.pop('conflict_prior_bias',None);plan.pop('prior_rule',None)
        plan.update(parent=str(prepared.relative_to(ROOT)), parent_source=str(checkpoint/'learner.zip'),
            selected_parent_checkpoint=str(checkpoint),source_run=str(source),
            seeds=seeds,training_bands=bands,prior_bias=6.,comparator_prior_bias=6.,
            heading_feedback='public_euler_heading_rate_v1',
            learner_feature_dim=44,public_observation_dim=39,
            intervention=dict(factor='public_heading_rate_observation',control='43features',value='44features',
                unchanged='projected original actor/critic/Adam/counters/teacher/prior6/PPO/48foes/quarter0.5/current public-rate reward/all other rewards/native physics/guns/actions'),
            comparison_scope='New public motion cue starts with zero actor/critic weights and Adam moments. Initial inference verified on consumed1609frame trace;cached672paired baseline reused with unchanged teacher/math. Tensor shapes and ZIP bytes change;not byte-identical or unseen validation.',
            parent_choice='Best complete672native prior6parent including unchanged comparator;provisional consumed-development choice',
            baseline_unique_games=672,baseline_panel_entries=800)
        inputs=[prepared,runner,Path(__file__).resolve(),ROOT/'tools/aim_prior_bias.py',
                ROOT/'tools/aim_cpu_rate_input_reporting.py',ROOT/'tools/aim_cpu_rate_input_export.py',
                ROOT/'experiments/aim_async/rate_input_worker.py',ROOT/'experiments/aim_async/rate_input_features.py',
                ROOT/'experiments/aim_async/rate_input_policy.py',
                ROOT/'tests/test_aim_rate_input.py',ROOT/'tests/test_prepare_aim_rate_input.py',
                source/'plan.json',source/'CYCLE_LOG_AUDIT.json',source/'TWO_REPEAT_REVIEW.json',
                source/'S46300_COMPLETED_REVIEW.json',source/'S46301_COMPLETED_REVIEW.json',
                diagnostic_path,trace_path,trace_path.parent/'summary.json',
                origin/'completion.json',completion,checkpoint/'READY.json',*paths]
        inputs.extend(p for p in (out/'initial_native').rglob('*') if p.is_file())
        inputs.extend(checkpoint/name for name in metadata['files'])
        plan['frozen_sha256'].update({str(p.relative_to(ROOT)): sha(p) for p in inputs})
        plan['config_id'] = identity({k:v for k,v in plan.items() if k not in ('config_id', 'frozen_sha256')})
        write(out/'plan.json', plan)
        for kind, cells in [('development', rows), ('monitor', monitor_rows)]:
            folder = out/'evaluations'/identity(dict(config_id=plan['config_id'], parent=metadata['checkpoint_id'], kind=kind))
            entry = dict(best, baseline=True, kind=kind, seed=seeds[0], additional_steps=0,
                source_additional_steps=metadata['additional_steps'], summary=profile(cells), win_gain=None,
                status='reused_zero_column_initial_prior6', actual_new_games=0,
                new_games_in_final_attempt=0, reused_cells_in_final_attempt=len(cells), wall_seconds=0.,
                checkpoint=str(checkpoint), reused_from_run=str(source), reused_from_evaluation=str(completion.parent),
                per_opponent={foe:profile([r for r in cells if r['opponent']==foe]) for foe in sorted({r['opponent'] for r in cells})},
                reuse_basis='Original columns and initializedAdam/counters preserved with zero new rate weights;initial greedy and both sampled modes verified on1609consumed trace frames. Teacher and original math retained;cached native cells reused under this zero-column equivalence,not a new44native tournament.')
            write(folder/'completion.json', entry); write(out/'candidates'/f'{folder.name}.json', entry)
            for row in cells:
                write(folder/'cells'/f"{identity({k:row[k] for k in ('opponent','seed','seat','action_seed')})}.json", row)
        write(out/'baseline_checkpoint.json', dict(path=str(checkpoint)))
        write(out/'parent_verification.json', dict(**change, restored=learner_state(restored),
            source_checkpoint=str(checkpoint), source_sha256=sha(checkpoint/'learner.zip'),
            prepared_sha256=sha(prepared), byte_identical=False,
            comparator_wins=measured['wins'], comparator_games=672, new_baseline_games=0,
            baseline_unique_games=672, baseline_panel_entries=800, initial_native_behavior_preserved=True))
        for band in bands:
            with (ROOT/f'runs/training_reservation_{band}.json').open('x') as f:
                json.dump(dict(run=str(out), start=band, stop_exclusive=band+1000000), f)
        write(out/'started.json', dict(time=time.time(), status='prepared_not_launched'))
        write(out/'goal_control.json', dict(status='prepared', goal_status='active', automatic_learning=True,
            publication=False, effective_objective=plan['objective'],
            next_action='One cumulative read-only code/artifact review then two public-rate-input44nativeCPU PPO repeats'))
        check_plan(original); check_plan(plan)
        print(json.dumps(dict(out=str(out), prepared=True, projected_tensor_Adam_counters_exact=True,
                              base_prior=6., heading_feedback=plan['heading_feedback'], comparator_wins=measured['wins'], new_baseline_games=0)))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); prepare(args.source.resolve(), args.out.resolve())
