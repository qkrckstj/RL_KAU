"""A one-scalar entropy comparison with exact preserved learner/cache reuse."""
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


def prepare(source, out, entropy):
    import math
    import psutil
    import torch
    from stable_baselines3 import PPO
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a fresh output under runs')
    if not math.isfinite(entropy) or not 0<entropy<=.01:
        raise ValueError('Use a small finite positive coefficient in0..0.01')
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
        if original['ppo']['ent_coef']!=0. or original.get('relative_potential_coefficient')!=1.:
            raise ValueError('Expected completed zero-entropy relative-reward control')
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
        seeds=[45600,45601];bands=[2500400000,2510400000]
        for file in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            r=read(file)
            if any(b<r['stop_exclusive'] and b+1000000>r['start'] for b in bands):raise ValueError('Reserved band: '+str(file))
        torch.set_num_threads(1);model=PPO.load(checkpoint/'learner.zip',device='cpu')
        def state(m):
            return dict(policy=tensor_digest(m.policy.state_dict()),optimizer=tensor_digest(m.policy.optimizer.state_dict()),decisions=m.num_timesteps,updates=m._n_updates)
        before=state(model)
        if before!=read(source/'parent_verification.json')['state'] or model.observation_space.shape!=(43,) or model.ent_coef!=0.:
            raise ValueError('Selected parent policy/Adam/counters/settings differ')
        out.mkdir(parents=True);model.ent_coef=entropy;prepared=out/'selected_parent.zip';model.save(prepared)
        restored=PPO.load(prepared,device='cpu')
        if state(restored)!=before or restored.ent_coef!=entropy:raise ValueError('One-scalar learner roundtrip differs')
        for key in ('n_epochs','batch_size','target_kl','max_grad_norm'):
            if getattr(restored,key)!=original['ppo'][key]:raise ValueError('Unexpected learner change: '+key)
        if restored.lr_schedule(1.)!=original['ppo']['learning_rate'] or restored.clip_range(1.)!=original['ppo']['clip_range']:
            raise ValueError('Learning/clip schedule changed')
        text=(source/'aim_cpu_runner.py').read_text()
        old="            write(out/f's{seed}_learning_complete.json',dict(time=time.time()))"
        new=old+"\n            from tools.aim_cpu_cycle_audit import audit, wait_for_seed_evaluations\n            wait_for_seed_evaluations(out,seed)\n            audit(out,seed)"
        if text.count(old)!=1:raise ValueError('Completed-repeat hook boundary differs')
        text=text.replace(old,new)
        old='            if meta and meta[\'additional_steps\']>=initial_limit:continue'
        new="            if meta and meta['additional_steps']>=initial_limit:\n                from tools.aim_cpu_cycle_audit import audit, wait_for_seed_evaluations\n                if not (out/f's{seed}_learning_complete.json').exists():\n                    write(out/f's{seed}_learning_complete.json',dict(time=(out/f's{seed}_result.json').stat().st_mtime,\n                        timing_basis='Saved result publication proxy after interruption; actual completion timestamp unavailable'))\n                wait_for_seed_evaluations(out,seed)\n                audit(out,seed)\n                continue"
        if text.count(old)!=1:raise ValueError('Resume hook boundary differs')
        text=text.replace(old,new);compile(text,str(out/'aim_cpu_runner.py'),'exec')
        runner=out/'aim_cpu_runner.py';runner.write_text(text)
        plan=copy.deepcopy(original);plan['ppo']['ent_coef']=entropy
        plan.update(parent=str(prepared.relative_to(ROOT)),parent_source=str((checkpoint/'learner.zip').relative_to(ROOT)),
            selected_parent_checkpoint=str(checkpoint),source_run=str(source),seeds=seeds,training_bands=bands,
            intervention=dict(factor='ppo.ent_coef',control=0.,value=entropy,unchanged='initial policy/Adam/counters, fixed aiming+relative reward,quarter starts,48foes and native evaluation'),
            comparison_scope='Adaptive two-seed entropy comparison, fresh RNGs; consumed development, not unseen final evidence.',
            completed_repeat_audit='Read every completed target-seed episode/CSV/paired cell before adaptive changes; no new matches.')
        inputs=[prepared,runner,Path(__file__).resolve(),ROOT/'tools/aim_cpu_cycle_audit.py',source/'plan.json',
                source/'parent_verification.json',source/'CYCLE_LOG_AUDIT.json',source/'TWO_REPEAT_REVIEW.json',
                source/'EXPLORATION_DIAGNOSIS.json',checkpoint/'READY.json',origin/'baseline_native_agreement.json']
        for folder,entry,rows,files in panels:inputs += [folder/'completion.json',*files]
        plan['frozen_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in inputs})
        plan['config_id']=identity({k:v for k,v in plan.items() if k not in ('config_id','frozen_sha256')});write(out/'plan.json',plan)
        for folder,entry,rows,files in panels:
            key=identity(dict(source=str(folder),config_id=plan['config_id']));reuse=reused_baseline(entry,folder,source,seeds[0])
            reuse['reuse_basis']='Initial actor/critic/Adam/counters exactly preserved; only training entropy scalar changes. Original43-feature baseline behavior/native evaluation and800paired cells unchanged.'
            write(out/'evaluations'/key/'completion.json',reuse);write(out/'candidates'/f'{key}.json',reuse)
            for row in rows:
                cell_id=identity({k:row[k] for k in ('opponent','seed','seat','action_seed')});write(out/'evaluations'/key/'cells'/f'{cell_id}.json',row)
        write(out/'baseline_checkpoint.json',dict(path=str(checkpoint)));write(out/'parent_verification.json',dict(exact=True,state=before,
            prepared_sha256=sha(prepared),source_sha256=sha(checkpoint/'learner.zip'),ent_coef_before=0.,ent_coef_after=entropy,
            reused_baseline_games=800,new_baseline_games=0,initial_behavior_basis='Policy tensors/class/feature layout and prior unchanged; entropy used only in PPO training loss.'))
        for band in bands:
            with (ROOT/f'runs/training_reservation_{band}.json').open('x') as f:json.dump(dict(run=str(out),start=band,stop_exclusive=band+1000000),f)
        write(out/'started.json',dict(time=time.time(),status='prepared_not_launched'))
        write(out/'goal_control.json',dict(status='prepared',automatic_learning=True,publication=False,effective_objective=plan['objective'],next_action='One read-only review, then two fixed-entropy repetitions with end-of-repeat audits.'))
        check_plan(original);check_plan(plan)
        print(json.dumps(dict(out=str(out),prepared=True,ent_coef=entropy,policy_Adam_counters_exact=True,reused_baseline_games=800,new_baseline_games=0)))
    finally:
        for lock in locks:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--entropy',type=float,default=.003)
    args=parser.parse_args();prepare(args.source.resolve(),args.out.resolve(),args.entropy)
