"""Continue a measured native parent without resetting policy or Adam state."""
import argparse
import copy
import fcntl
import json
import shutil
import time
from pathlib import Path
from experiments.rear_async.storage import identity,read,sha,verify,write
from tools.prepare_rear_cpu_entropy import tensor_digest
from tools.rear_async_cpu import ROOT,check_plan


def hard_monitor(foes,reference):
    ranked=sorted(foes,key=lambda q:(reference[q['id']]['wins'],q['id']))
    if len(ranked)<8:raise ValueError('Need eight representative monitor foes')
    return ranked[:8],4,[4900,4901]


def prepare(source,out):
    import psutil
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Use fresh output under runs')
    if not (source/'training_complete.json').exists() or (source/'training_failure.json').exists():
        raise RuntimeError('Source not successfully terminal')
    if read(source/'evaluation_complete.json')['status']!='queue_drained' or list((source/'evaluations').glob('*/blocked.json')):
        raise RuntimeError('Source evaluations incomplete or blocked')
    for proc in psutil.process_iter(['cmdline','status']):
        args=proc.info['cmdline'] or []
        if proc.info['status']!=psutil.STATUS_ZOMBIE and any(Path(a).name=='aim_cpu_runner.py' or a=='tools.rear_async_cpu' for a in args) and any(a in args for a in ('launch','evaluate')):
            raise RuntimeError('Existing native trainer/evaluator: '+str(proc.pid))
    if psutil.virtual_memory().available<10*2**30:raise MemoryError('Need10GiB available')
    locks=[]
    try:
        for name in ('training.lock','evaluation.lock','launch.lock'):
            lock=(source/name).open('a');locks.append(lock);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=read(source/'plan.json');check_plan(original)
        review=read(source/'TWO_REPEAT_REVIEW.json');audit=read(source/'CYCLE_LOG_AUDIT.json')
        if review['status']!='two_repeats_complete_analyzed':raise ValueError('Source not fully analyzed')
        if set(audit['training'])!={str(s) for s in original['seeds']}:raise ValueError('Completed audits missing')
        for seed in original['seeds']:
            t=audit['training'][str(seed)]
            if t['all_episodes']['games']!=read(source/f's{seed}_result.json')['training_games'] or t['nonfinite_ppo_fields']:
                raise ValueError('Completed training audit differs or nonfinite')
        if original['control_guidance_range']!=12000. or original['control_guidance_coefficient']!=.05 or original['ppo']['ent_coef']!=.003:
            raise ValueError('Expected unchanged12km/0.05/entropy0.003 control')
        entries=[(p,read(p)) for p in (source/'evaluations').glob('*/completion.json')]
        reference=next(e for _,e in entries if e['baseline'] and e['kind']=='development')
        eligible=[(p,e) for p,e in entries if not e['baseline'] and e['kind']=='development']
        completion,best=max(eligible,key=lambda q:(q[1]['summary']['wins'],q[1]['summary']['damage_dealt']-q[1]['summary']['damage_received']))
        if best['summary']['games']!=672 or best['summary']['wins']<=reference['summary']['wins']:
            raise ValueError('No better measured native parent')
        checkpoint=Path(best['checkpoint']).resolve()
        if not checkpoint.is_relative_to(source.resolve()/'checkpoints'):raise ValueError('Parent outside source checkpoints')
        metadata=verify(checkpoint)
        if metadata['config_id']!=original['config_id'] or metadata['checkpoint_id']!=best['checkpoint_id'] or metadata['additional_steps']!=best['additional_steps']:
            raise ValueError('Evaluated parent identity differs')
        if verify(completion.parent/'input')!=metadata:raise ValueError('Evaluated input differs from parent')
        paths=sorted(completion.parent.glob('cells/*.json'));rows=[read(p) for p in paths]
        keys={tuple(q[k] for k in ('opponent','seed','seat','action_seed')) for q in rows}
        expected={(f['id'],original['development_band']+i,seat,a) for f in original['development_opponents'] for i in range(8) for seat in ('red','blue') for a in (4900,4901)}
        if len(rows)!=672 or keys!=expected or sum(q['result']=='win' for q in rows)!=best['summary']['wins']:
            raise ValueError('Evaluated parent cells incomplete or unpaired')
        from experiments.aim_async.evaluate import profile
        measured=profile(rows)
        for key in ('games','wins','damage_dealt','damage_received'):
            if abs(measured[key]-best['summary'][key])>1e-12:raise ValueError('Parent profile differs')
        monitor,ics,samplers=hard_monitor(original['development_opponents'],reference['per_opponent'])
        monitor_ids={f['id'] for f in monitor}
        monitor_rows=[q for q in rows if q['opponent'] in monitor_ids and q['seed']<original['development_band']+ics]
        if len(monitor_rows)!=128:raise ValueError('Hard monitor subset differs')
        seeds=[46000,46001];bands=[2580400000,2590400000]
        for path in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            reservation=read(path)
            if any(b<reservation['stop_exclusive'] and b+1000000>reservation['start'] for b in bands):
                raise ValueError('Reserved training band: '+str(path))
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1);model=PPO.load(checkpoint/'learner.zip',device='cpu')
        state=dict(policy=tensor_digest(model.policy.state_dict()),optimizer=tensor_digest(model.policy.optimizer.state_dict()),decisions=model.num_timesteps,updates=model._n_updates)
        if model.observation_space.shape!=(43,) or model.action_space.n!=9 or model.policy.prior_bias!=6 or model.n_envs!=32:
            raise ValueError('Parent policy/prior/spaces differ')
        for key in ('n_steps','n_epochs','batch_size','gamma','gae_lambda','target_kl','ent_coef','max_grad_norm'):
            if getattr(model,key)!=original['ppo'][key]:raise ValueError('Parent PPO setting differs: '+key)
        if model.lr_schedule(1.)!=original['ppo']['learning_rate'] or model.clip_range(1.)!=original['ppo']['clip_range']:
            raise ValueError('Parent optimizer schedules differ')
        if model.num_timesteps!=metadata['prefix_steps']+metadata['additional_steps']:raise ValueError('Parent decision count differs')
        code=(source/'aim_cpu_runner.py').read_text()
        old="replicas=[4900] if job['kind']=='monitor' else [4900,4901]"
        if code.count(old)!=1:raise ValueError('Runner sampler boundary differs')
        code=code.replace(old,"replicas=plan['monitor_action_seeds'] if job['kind']=='monitor' else [4900,4901]")
        old="""                    else:
                        if (out/'STOP').exists():raise InterruptedError('STOP marker')
                        row=dict(cell(own,opponent,band+i,seat),action_seed=action_seed)
                        write(file,row);new+=1
"""
        new="""                    else:
                        if (out/'STOP').exists():raise InterruptedError('STOP marker')
                        from experiments.aim_async.shared_eval_cache import shared_cell
                        key_fields=dict(opponent=opponent['id'],seed=band+i,seat=seat,action_seed=action_seed)
                        reused=shared_cell(out,job,m,key_fields)
                        if reused is None:
                            row=dict(cell(own,opponent,band+i,seat),action_seed=action_seed)
                            write(file,row);new+=1
                        else:
                            row,proof=reused
                            manifest=read(folder/'reuse_manifest.json') if (folder/'reuse_manifest.json').exists() else {}
                            manifest[key]=proof;write(folder/'reuse_manifest.json',manifest);write(file,row)
"""
        if code.count(old)!=1:raise ValueError('Runner native cell boundary differs')
        code=code.replace(old,new)
        old='actual_new_games=len(rows),new_games_in_final_attempt=new,reused_cells_in_final_attempt=len(rows)-new,wall_seconds='
        if code.count(old)!=1:raise ValueError('Runner count boundary differs')
        code=code.replace(old,'**reuse_counts(folder,rows,new),wall_seconds=')
        code=code.replace('    entry=dict(checkpoint_id=m[',"    from experiments.aim_async.shared_eval_cache import reuse_counts\n    entry=dict(checkpoint_id=m[")
        code=code.replace('from tools.aim_cpu_reporting import update','from tools.aim_cpu_continuation_reporting import update')
        compile(code,'aim_cpu_runner.py','exec')
        out.mkdir(parents=True);prepared=out/'selected_parent.zip';shutil.copyfile(checkpoint/'learner.zip',prepared)
        if sha(prepared)!=sha(checkpoint/'learner.zip'):raise ValueError('Copied parent learner differs')
        runner=out/'aim_cpu_runner.py';runner.write_text(code)
        plan=copy.deepcopy(original);plan.update(parent=str(prepared.relative_to(ROOT)),parent_source=str(checkpoint/'learner.zip'),selected_parent_checkpoint=str(checkpoint),source_run=str(source),seeds=seeds,training_bands=bands,monitor_opponents=monitor,monitor_ics=ics,monitor_action_seeds=samplers,monitor_band=original['development_band'],original_reference_checkpoint=read(source/'baseline_checkpoint.json')['path'],original_reference_wins=reference['summary']['wins'],parent_choice='Provisional best measured672native wins;not final generalization confirmation',baseline_unique_games=672,baseline_panel_entries=800,comparison_scope='Unchanged reward/PPO continuation of best measured native parent;hard monitor changes screen coverage only;full672acceptance unchanged',intervention=dict(factor='best_measured_parent_continuation',parent_wins=best['summary']['wins'],original_wins=reference['summary']['wins'],unchanged='native physics/guns/actions,43features,PPO,48trainingfoes,quarter0.5,12kmreward'),budget_policy='Combat retention/trends and paired native672win/HP gain gate bounded extension;rear scores never gate continuation.')
        inputs=[prepared,runner,Path(__file__).resolve(),ROOT/'experiments/aim_async/shared_eval_cache.py',ROOT/'tools/aim_cpu_continuation_reporting.py',source/'plan.json',source/'CYCLE_LOG_AUDIT.json',source/'TWO_REPEAT_REVIEW.json',completion,checkpoint/'READY.json',*paths]
        inputs.extend(checkpoint/name for name in metadata['files'])
        plan['frozen_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in inputs});plan['config_id']=identity({k:v for k,v in plan.items() if k not in ('config_id','frozen_sha256')});write(out/'plan.json',plan)
        for kind,cells in [('development',rows),('monitor',monitor_rows)]:
            folder=out/'evaluations'/identity(dict(config_id=plan['config_id'],parent=metadata['checkpoint_id'],kind=kind))
            entry=dict(best,baseline=True,kind=kind,seed=seeds[0],additional_steps=0,source_additional_steps=best['additional_steps'],summary=profile(cells),win_gain=None,status='reused_provisional_best_parent_baseline',actual_new_games=0,new_games_in_final_attempt=0,reused_cells_in_final_attempt=len(cells),wall_seconds=0.,reused_from_run=str(source),reused_from_evaluation=str(completion.parent),per_opponent={foe:profile([q for q in cells if q['opponent']==foe]) for foe in sorted({q['opponent'] for q in cells})},reuse_basis='Byte-identical evaluated parent;672unique native matches,monitor128subset;no new baseline games')
            write(folder/'completion.json',entry);write(out/'candidates'/f'{folder.name}.json',entry)
            for row in cells:write(folder/'cells'/f"{identity({k:row[k] for k in ('opponent','seed','seat','action_seed')})}.json",row)
        write(out/'baseline_checkpoint.json',dict(path=str(checkpoint)))
        write(out/'parent_verification.json',dict(exact=True,state=state,prepared_sha256=sha(prepared),source_sha256=sha(checkpoint/'learner.zip'),source_checkpoint=str(checkpoint),parent_wins=best['summary']['wins'],original_reference_wins=reference['summary']['wins'],baseline_unique_games=672,baseline_panel_entries=800,new_baseline_games=0,provisional=True))
        for band in bands:
            with (ROOT/f'runs/training_reservation_{band}.json').open('x') as f:json.dump(dict(run=str(out),start=band,stop_exclusive=band+1000000),f)
        write(out/'started.json',dict(time=time.time(),status='prepared_not_launched'))
        write(out/'goal_control.json',dict(status='prepared',goal_status='active',automatic_learning=True,publication=False,effective_objective=plan['objective'],next_action='One read-only review then unchanged-reward best-parent PPO continuation'))
        check_plan(original);check_plan(plan)
        print(json.dumps(dict(out=str(out),prepared=True,parent_wins=best['summary']['wins'],parent_bytes_Adam_counters_exact=True,baseline_unique_games=672,baseline_panel_entries=800,new_baseline_games=0)))
    finally:
        for lock in locks:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();prepare(args.source.resolve(),args.out.resolve())
