"""Prepare an earlier guidance range comparison with exact native lineage."""
import argparse
import copy
import fcntl
import json
import shutil
import time
from pathlib import Path
from experiments.rear_async.storage import read, write, sha, identity, verify
from tools.prepare_aim_mixed_start import validate_cached_panel
from tools.prepare_rear_cpu_selected_parent import reused_baseline
from tools.prepare_rear_cpu_entropy import tensor_digest
from tools.rear_async_cpu import ROOT, check_plan


def prepare(source, out, range_limit):
    import math
    import psutil
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a fresh output under runs')
    if not math.isfinite(range_limit) or range_limit!=12000.:
        raise ValueError('Use guidance range12000m for this one-factor comparison')
    coefficient=.05
    if not (source/'training_complete.json').exists() or (source/'training_failure.json').exists():
        raise RuntimeError('Source not successfully terminal')
    if read(source/'evaluation_complete.json')['status']!='queue_drained' or list((source/'evaluations').glob('*/blocked.json')):
        raise RuntimeError('Source evaluations incomplete or blocked')
    for process in psutil.process_iter(['cmdline','status']):
        args=process.info['cmdline'] or []
        if process.info['status']!=psutil.STATUS_ZOMBIE and any(
                Path(a).name=='aim_cpu_runner.py' or a=='tools.rear_async_cpu' for a in args) and any(a in args for a in ('launch','evaluate')):
            raise RuntimeError('Existing native trainer/evaluator: '+str(process.pid))
    if psutil.virtual_memory().available<10*2**30:raise MemoryError('Need10GiB available')
    locks=[]
    try:
        for name in ('training.lock','evaluation.lock','launch.lock'):
            lock=(source/name).open('a');locks.append(lock);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=read(source/'plan.json');check_plan(original)
        latest=read(source/'TWO_REPEAT_REVIEW.json')
        if latest['promotion'] or latest['status']!='two_repeats_complete_analyzed':
            raise ValueError('Source not fully analyzed or contains improved endpoint')
        diagnostic=read(source/'diagnostics/guidance_signal/summary.json')
        if not diagnostic['exact_cached_reproduction'] or diagnostic['diagnostic_games']!=1:
            raise ValueError('Expected exact consumed native guidance diagnostic')
        if (original['ppo']['ent_coef']!=.003 or original.get('relative_potential_coefficient')!=1.
                or original['aim_coefficients']!=dict(potential=.05,damage=.5,received=.5)
                or original.get('control_guidance_coefficient')!=.05
                or original.get('control_guidance_range',6000.)!=6000.):
            raise ValueError('Expected completed fixed guidance0.05/6km control')
        entries=[read(f) for f in (source/'evaluations').glob('*/completion.json')]
        baseline=next(e for e in entries if e['baseline'] and e['kind']=='development')
        endpoints=[e for e in entries if not e['baseline'] and e['kind']=='development']
        if len(endpoints)!=2 or any(e['summary']['wins']>baseline['summary']['wins'] for e in endpoints):
            raise ValueError('Review improved or incomplete candidates before baseline continuation')
        if any(read(source/f's{s}_budget_decision.json')['extend'] for s in original['seeds']):
            raise ValueError('Unexpected extended control; review first')
        audit=read(source/'CYCLE_LOG_AUDIT.json')
        if set(audit['training'])!={str(s) for s in original['seeds']}:
            raise ValueError('Completed repeat audit missing')
        for s in original['seeds']:
            if audit['training'][str(s)]['all_episodes']['games']!=read(source/f's{s}_result.json')['training_games']:
                raise ValueError('Audited episode count differs')
        checkpoint=Path(read(source/'baseline_checkpoint.json')['path']);meta=verify(checkpoint)
        if not meta['baseline']:raise ValueError('Expected preserved baseline')
        origin=checkpoint.parent.parent;origin_plan=read(origin/'plan.json');check_plan(origin_plan)
        if meta['config_id']!=origin_plan['config_id']:raise ValueError('Baseline origin identity differs')
        agreement=read(origin/'baseline_native_agreement.json')
        if not agreement['exact'] or agreement['games']!=128:raise ValueError('No exact native baseline lineage')
        origin_monitors=[f.parent for f in (origin/'evaluations').glob('*/completion.json')
                         if read(f)['baseline'] and read(f)['kind']=='monitor' and read(f)['checkpoint_id']==meta['checkpoint_id']]
        if len(origin_monitors)!=1 or verify(origin_monitors[0]/'input')!=meta:
            raise ValueError('Originally evaluated baseline differs')
        panels=[]
        for kind in ('monitor','development'):
            folders=[f.parent for f in (source/'evaluations').glob('*/completion.json') if read(f)['baseline'] and read(f)['kind']==kind]
            if len(folders)!=1:raise ValueError('Ambiguous source baseline')
            entry,rows,files=validate_cached_panel(original,folders[0])
            if entry['checkpoint_id']!=meta['checkpoint_id']:raise ValueError('Inherited baseline identity differs')
            panels.append((folders[0],entry,rows,files))
        seeds=[45900,45901];bands=[2560400000,2570400000]
        for file in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            r=read(file)
            if any(b<r['stop_exclusive'] and b+1000000>r['start'] for b in bands):raise ValueError('Reserved band: '+str(file))
        import torch
        from stable_baselines3 import PPO
        learner=ROOT/original['parent']
        if sha(learner)!=read(source/'parent_verification.json')['prepared_sha256']:
            raise ValueError('Serialized original control parent identity differs')
        torch.set_num_threads(1);model=PPO.load(learner,device='cpu')
        def state(m):
            return dict(policy=tensor_digest(m.policy.state_dict()),optimizer=tensor_digest(m.policy.optimizer.state_dict()),decisions=m.num_timesteps,updates=m._n_updates)
        before=state(model)
        if before!=read(source/'parent_verification.json')['state'] or model.observation_space.shape!=(43,) or model.ent_coef!=.003:
            raise ValueError('Selected parent policy/Adam/counters/settings differ')
        if model.gamma!=original['ppo']['gamma'] or model.gae_lambda!=original['ppo']['gae_lambda']:
            raise ValueError('Discount/advantage settings differ')
        if model.n_steps!=640 or model.n_envs!=32 or model.policy.prior_bias!=6 or model.action_space.n!=9:
            raise ValueError('Rollout/prior/action settings differ')
        out.mkdir(parents=True);prepared=out/'selected_parent.zip';shutil.copyfile(learner,prepared)
        if sha(prepared)!=sha(learner):raise ValueError('Serialized control learner copy differs')
        restored=model
        for key in ('n_epochs','batch_size','target_kl','max_grad_norm'):
            if getattr(restored,key)!=original['ppo'][key]:raise ValueError('Unexpected learner change: '+key)
        if restored.lr_schedule(1.)!=original['ppo']['learning_rate'] or restored.clip_range(1.)!=original['ppo']['clip_range']:
            raise ValueError('Learning/clip schedule changed')
        runner=out/'aim_cpu_runner.py';code=(source/'aim_cpu_runner.py').read_text()
        old='from experiments.aim_async.control_guidance_worker import make_worker'
        if code.count(old)!=1:raise ValueError('Original worker boundary differs')
        code=code.replace(old,'from experiments.aim_async.guidance_range_worker import make_worker')
        compile(code,str(runner),'exec');runner.write_text(code)
        plan=copy.deepcopy(original);plan['control_guidance_range']=range_limit
        plan.update(parent=str(prepared.relative_to(ROOT)),parent_source=str(learner.relative_to(ROOT)),
            selected_parent_checkpoint=str(checkpoint),source_run=str(source),diagnostic_evidence_run=str(source),seeds=seeds,training_bands=bands,
            intervention=dict(factor='control_guidance_range',control=6000.,value=range_limit,unchanged='initial policy/critic/Adam/counters,entropy0.003,received0.5,control guidance0.05 and formulas,all original rewards/quarter starts/48foes/native evaluation'),
            comparison_scope='Adaptive two-seed early-guidance-range comparison versus completed guidance0.05/6km control;fresh RNGs,consumed development,not unseen final evidence.',
            completed_repeat_audit=original['completed_repeat_audit'])
        inputs=[prepared,runner,Path(__file__).resolve(),ROOT/'tools/aim_cpu_cycle_audit.py',source/'plan.json',
                source/'parent_verification.json',source/'CYCLE_LOG_AUDIT.json',source/'TWO_REPEAT_REVIEW.json',
                learner,checkpoint/'READY.json',origin/'baseline_native_agreement.json',
                ROOT/'experiments/aim_async/guidance_range_worker.py',
                source/'diagnostics/guidance_signal/summary.json',source/'diagnostics/guidance_signal/trace.jsonl',
                source/'diagnose_guidance_signal.py']
        for folder,entry,rows,files in panels:inputs += [folder/'completion.json',*files]
        plan['frozen_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in inputs})
        plan['config_id']=identity({k:v for k,v in plan.items() if k not in ('config_id','frozen_sha256')});write(out/'plan.json',plan)
        for folder,entry,rows,files in panels:
            key=identity(dict(source=str(folder),config_id=plan['config_id']));reuse=reused_baseline(entry,folder,source,seeds[0])
            reuse['reuse_basis']='Serialized initial control learner is byte-identical,including actor/critic/Adam/counters/entropy. Only training guidance range changes;native baseline inference and800paired cells unchanged.'
            write(out/'evaluations'/key/'completion.json',reuse);write(out/'candidates'/f'{key}.json',reuse)
            for row in rows:
                cell_id=identity({k:row[k] for k in ('opponent','seed','seat','action_seed')});write(out/'evaluations'/key/'cells'/f'{cell_id}.json',row)
        write(out/'baseline_checkpoint.json',dict(path=str(checkpoint)));write(out/'parent_verification.json',dict(exact=True,state=before,
            prepared_sha256=sha(prepared),source_sha256=sha(learner),ent_coef=.003,control_guidance_coefficient=coefficient,control_guidance_range_before=6000.,control_guidance_range_after=range_limit,
            reused_baseline_games=800,new_baseline_games=0,initial_behavior_basis='Byte-identical original control learner;training-only potential cannot alter initial inference or native baseline cells.'))
        for band in bands:
            with (ROOT/f'runs/training_reservation_{band}.json').open('x') as f:json.dump(dict(run=str(out),start=band,stop_exclusive=band+1000000),f)
        write(out/'started.json',dict(time=time.time(),status='prepared_not_launched'))
        write(out/'goal_control.json',dict(status='prepared',automatic_learning=True,publication=False,effective_objective=plan['objective'],next_action='One read-only review,then two fixed early-guidance-range repetitions with completed per-seed log audits.'))
        check_plan(original);check_plan(plan)
        print(json.dumps(dict(out=str(out),prepared=True,guidance_coefficient=coefficient,guidance_range=range_limit,learner_bytes_policy_Adam_counters_exact=True,reused_baseline_games=800,new_baseline_games=0)))
    finally:
        for lock in locks:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--range-limit',type=float,default=12000.)
    args=parser.parse_args();prepare(args.source.resolve(),args.out.resolve(),args.range_limit)
