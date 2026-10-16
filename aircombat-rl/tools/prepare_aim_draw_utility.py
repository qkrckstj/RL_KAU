"""Prepare a separate terminal draw-utility study after completed input trials."""
import argparse
import copy
import fcntl
import json
import shutil
import time
from pathlib import Path
from experiments.rear_async.storage import identity, read, sha, verify, write
from tools.rear_async_cpu import ROOT, check_plan
from tools.aim_prior_bias import learner_state
from tools.prepare_aim_heading_rate import select_parent, comparator_origin


def prepare(source, out):
    import psutil
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Use fresh output under runs')
    if not (source/'training_complete.json').exists() or (source/'training_failure.json').exists():
        raise RuntimeError('Source not successfully terminal')
    if read(source/'evaluation_complete.json')['status'] != 'queue_drained' or list((source/'evaluations').glob('*/blocked.json')):
        raise RuntimeError('Source evaluations incomplete or blocked')
    for proc in psutil.process_iter(['cmdline','status']):
        args = proc.info['cmdline'] or []
        if (proc.info['status'] != psutil.STATUS_ZOMBIE
                and any(Path(a).name == 'aim_cpu_runner.py' or a == 'tools.rear_async_cpu' for a in args)
                and any(a in args for a in ('launch','evaluate'))):
            raise RuntimeError('Existing native trainer/evaluator: '+str(proc.pid))
    if psutil.virtual_memory().available < 10*2**30:
        raise MemoryError('Need10GiB available')
    locks = []
    try:
        for name in ('training.lock','evaluation.lock','launch.lock'):
            lock = (source/name).open('a'); locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        original = read(source/'plan.json'); check_plan(original)
        review, audit = read(source/'TWO_REPEAT_REVIEW.json'), read(source/'CYCLE_LOG_AUDIT.json')
        if review['status'] != 'two_repeats_complete_analyzed' or set(audit['training']) != {str(s) for s in original['seeds']}:
            raise ValueError('Source repeats not completely analyzed')
        for seed in original['seeds']:
            a = audit['training'][str(seed)]
            if a['all_episodes']['games'] != read(source/f's{seed}_result.json')['training_games'] or a['nonfinite_ppo_fields']:
                raise ValueError('Completed source audit differs or nonfinite')
        if (original['intervention']['factor'] != 'public_heading_command_memory'
                or original['heading_feedback'] != 'public_euler_heading_rate_v1'
                or original['control_guidance_range'] != 12000.
                or original['control_guidance_coefficient'] != .05
                or original['prior_bias'] != 6. or original['comparator_prior_bias'] != 6.
                or original['processes'] != 8 or original['workers'] != 32
                or original['monitor_ics'] != 4 or original['development_ics'] != 8
                or original['monitor_action_seeds'] != [4900,4901]
                or original['monitor_band'] != original['development_band']):
            raise ValueError('Expected completed native memory-input study')
        entries = [(p,read(p)) for p in (source/'evaluations').glob('*/completion.json')]
        completion, best = select_parent(entries)
        endpoints = [e for _,e in entries if not e['baseline'] and e['kind'] == 'development']
        if (not best['baseline'] or len(endpoints) != 2
                or any(e['summary']['wins'] >= best['summary']['wins'] for e in endpoints)):
            raise ValueError('Memory candidates did not both regress; reconsider next factor')
        checkpoint = Path(best['checkpoint']).resolve(); metadata = verify(checkpoint)
        origin = comparator_origin(completion,best)
        if checkpoint != Path(read(source/'baseline_checkpoint.json')['path']).resolve() or verify(origin/'input') != metadata:
            raise ValueError('Original comparator checkpoint differs')
        paths = sorted(completion.parent.glob('cells/*.json')); rows = [read(p) for p in paths]
        expected = {(f['id'],original['development_band']+i,seat,sampler)
            for f in original['development_opponents'] for i in range(8)
            for seat in ('red','blue') for sampler in (4900,4901)}
        indexed = lambda rr: {tuple(r[k] for k in ('opponent','seed','seat','action_seed')):r for r in rr}
        if (len(rows) != 672 or set(indexed(rows)) != expected
                or indexed(rows) != indexed([read(p) for p in origin.glob('cells/*.json')])):
            raise ValueError('Cached comparator cells differ from original native results')
        from experiments.aim_async.evaluate import profile
        measured = profile(rows)
        for key in ('games','wins','damage_dealt','damage_received','first_shot_advantage_rate'):
            if abs(measured[key]-best['summary'][key]) > 1e-12:
                raise ValueError('Comparator profile differs')
        monitor_ids = {f['id'] for f in original['monitor_opponents']}
        monitor_rows = [r for r in rows if r['opponent'] in monitor_ids and r['seed'] < original['development_band']+4]
        if len(monitor_rows) != 128:
            raise ValueError('Monitor subset differs')
        seeds, bands = [46600,46601], [2700400000,2710400000]
        for path in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            reservation = read(path)
            if any(b < reservation['stop_exclusive'] and b+1000000 > reservation['start'] for b in bands):
                raise ValueError('Reserved training band: '+str(path))
        import torch
        from stable_baselines3 import PPO
        from experiments.aim_async.policy import AimPriorPolicy
        torch.set_num_threads(1); model = PPO.load(checkpoint/'learner.zip',device='cpu')
        if type(model.policy) is not AimPriorPolicy or model.policy.prior_bias != 6. or model.observation_space.shape != (43,) or model.n_envs != 32:
            raise ValueError('Expected original43feature prior6learner')
        for key in ('n_steps','n_epochs','batch_size','gamma','gae_lambda','target_kl','ent_coef','max_grad_norm'):
            if getattr(model,key) != original['ppo'][key]:
                raise ValueError('Parent PPO setting differs: '+key)
        if (model.lr_schedule(1.) != original['ppo']['learning_rate'] or model.clip_range(1.) != original['ppo']['clip_range']
                or model.num_timesteps != metadata['total_steps']):
            raise ValueError('Parent schedules or counters differ')
        code = (source/'aim_cpu_runner.py').read_text()
        substitutions = [
            ('from experiments.aim_async.heading_memory_worker import make_worker','from experiments.aim_async.draw_utility_worker import make_worker'),
            ('from experiments.aim_async.heading_memory_policy import HeadingMemoryPriorPolicy','from experiments.aim_async.policy import AimPriorPolicy'),
            ("assert type(model.policy) is HeadingMemoryPriorPolicy and model.observation_space.shape==(44,) and model.policy.prior_bias==plan['prior_bias'] and model.gamma==.999 and model.gae_lambda==.99", "assert type(model.policy) is AimPriorPolicy and model.observation_space.shape==(43,) and model.policy.prior_bias==plan['prior_bias'] and model.gamma==.999 and model.gae_lambda==.99"),
            ('from tools.aim_cpu_heading_memory_export import export','from tools.aim_cpu_export import export'),
            ('from tools.aim_cpu_heading_memory_action_mode import build_variant','from tools.league_residual_action_mode import build_variant'),
            ('from tools.aim_cpu_heading_memory_reporting import update','from tools.aim_cpu_draw_utility_reporting import update')]
        for old,new in substitutions:
            if code.count(old) != 1:
                raise ValueError('Runner boundary differs: '+old)
            code = code.replace(old,new)
        compile(code,'aim_cpu_runner.py','exec')
        before = learner_state(model)
        out.mkdir(parents=True); prepared = out/'draw_utility_parent.zip'
        shutil.copyfile(checkpoint/'learner.zip',prepared)
        restored = PPO.load(prepared,device='cpu')
        if learner_state(restored) != before or sha(prepared) != sha(checkpoint/'learner.zip'):
            raise ValueError('Copied initial learner bytes/NN/Adam/counters differ')
        runner = out/'aim_cpu_runner.py'; runner.write_text(code)
        plan = copy.deepcopy(original); plan.pop('heading_memory_contract',None)
        plan.update(parent=str(prepared.relative_to(ROOT)),parent_source=str(checkpoint/'learner.zip'),
            selected_parent_checkpoint=str(checkpoint),source_run=str(source),seeds=seeds,training_bands=bands,
            learner_feature_dim=43,public_observation_dim=39,terminal_draw_penalty=.25,
            scope='Fixed training terminal draw utility from preserved original43feature498reference.',
            intervention=dict(factor='terminal_draw_utility',control=0.,value=-.25,
                unchanged='initial learner bytes/NN/Adam/counters/43features/prior6/teacher/PPO/48foes/quarter0.5/public-rate guidance and all other rewards/native physics/guns/actions/health adjudication'),
            comparison_scope='Learning-only draw utility. Original43feature reference ZIP byte-identical; cached672paired native baseline reused. Previous43public-rate controls494/493 used different training seeds.',
            parent_choice='Original43feature best complete672native reference; provisional consumed-development choice')
        inputs = [prepared,runner,Path(__file__).resolve(),ROOT/'tools/aim_cpu_draw_utility_reporting.py',
            ROOT/'experiments/aim_async/draw_utility_worker.py',ROOT/'tests/test_aim_draw_utility.py',
            source/'plan.json',source/'CYCLE_LOG_AUDIT.json',source/'TWO_REPEAT_REVIEW.json',
            source/'S46500_COMPLETED_REVIEW.json',source/'S46501_COMPLETED_REVIEW.json',
            source/'DRAW_UTILITY_NEXT_STUDY.md',source/'CONDITIONAL_POLICY_DRIFT_DIAGNOSIS.json',
            source/'NEW_CUE_OPTIMIZER_AGE_DIAGNOSIS.json',source/'REPEAT_RESULTS_KO.md',
            ROOT/'tools/prepare_aim_heading_rate.py',completion,origin/'completion.json',checkpoint/'READY.json',*paths]
        inputs.extend(checkpoint/name for name in metadata['files'])
        plan['frozen_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in inputs})
        plan['config_id'] = identity({k:v for k,v in plan.items() if k not in ('config_id','frozen_sha256')})
        write(out/'plan.json',plan)
        for kind,cells in [('development',rows),('monitor',monitor_rows)]:
            folder = out/'evaluations'/identity(dict(config_id=plan['config_id'],parent=metadata['checkpoint_id'],kind=kind))
            entry = dict(best,baseline=True,kind=kind,seed=seeds[0],additional_steps=0,
                source_additional_steps=metadata['additional_steps'],summary=profile(cells),win_gain=None,
                status='reused_byte_identical_initial_prior6',actual_new_games=0,new_games_in_final_attempt=0,
                reused_cells_in_final_attempt=len(cells),wall_seconds=0.,checkpoint=str(checkpoint),
                reused_from_run=str(source),reused_from_evaluation=str(completion.parent),
                per_opponent={f:profile([r for r in cells if r['opponent']==f]) for f in sorted({r['opponent'] for r in cells})},
                reuse_basis='Original43feature learner ZIP byte-identical including NN/initializedAdam/counters. Terminal draw utility changes learning only; native initial inference remains identical.')
            write(folder/'completion.json',entry);write(out/'candidates'/f'{folder.name}.json',entry)
            for row in cells:
                write(folder/'cells'/f"{identity({k:row[k] for k in ('opponent','seed','seat','action_seed')})}.json",row)
        write(out/'baseline_checkpoint.json',dict(path=str(checkpoint)))
        write(out/'parent_verification.json',dict(before=before,after=before,restored=learner_state(restored),
            state_exact=True,source_checkpoint=str(checkpoint),source_sha256=sha(checkpoint/'learner.zip'),
            prepared_sha256=sha(prepared),byte_identical=True,policy_behavior_unchanged=True,
            comparator_wins=measured['wins'],comparator_games=672,new_baseline_games=0,
            baseline_unique_games=672,baseline_panel_entries=800,initial_native_behavior_preserved=True))
        for band in bands:
            with (ROOT/f'runs/training_reservation_{band}.json').open('x') as f:
                json.dump(dict(run=str(out),start=band,stop_exclusive=band+1000000),f)
        write(out/'previous_league_state.json',read(ROOT.parent/'docs/LEAGUE_STATE.json'))
        write(out/'started.json',dict(time=time.time(),status='prepared_not_launched'))
        write(out/'goal_control.json',dict(status='prepared',goal_status='active',automatic_learning=True,
            publication=False,effective_objective=plan['objective'],next_action='One cumulative read-only review,then two fixed draw-utility CPU PPO repeats.'))
        check_plan(original);check_plan(plan)
        print(json.dumps(dict(out=str(out),prepared=True,byte_identical=True,comparator_wins=measured['wins'],new_baseline_games=0,frozen_files=len(plan['frozen_sha256']))))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args();prepare(args.source.resolve(),args.out.resolve())
