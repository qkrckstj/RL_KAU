"""Restore one general shaping component from the preserved aiming baseline."""
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


def prepare(source, assessment, out):
    import psutil
    import torch
    from stable_baselines3 import PPO
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a fresh output under runs')
    for run in (source, assessment):
        if not (run/'training_complete.json').exists() or (run/'training_failure.json').exists():
            raise RuntimeError('Predecessor not successfully terminal')
        if read(run/'evaluation_complete.json')['status'] != 'queue_drained':
            raise RuntimeError('Predecessor evaluation not complete')
        if list((run/'evaluations').glob('*/blocked.json')):
            raise RuntimeError('Predecessor evaluation blocked')
    for process in psutil.process_iter(['cmdline','status']):
        args=process.info['cmdline'] or []
        if process.info['status'] != psutil.STATUS_ZOMBIE and any(
                Path(a).name=='aim_cpu_runner.py' for a in args) and any(a in args for a in ('launch','evaluate')):
            raise RuntimeError('Native trainer/evaluator still live: '+str(process.pid))
    if psutil.virtual_memory().available < 10*2**30:
        raise MemoryError('Need10GiB available')
    locks=[]
    try:
        for run,name in ((source,'training.lock'),(source,'evaluation.lock'),(assessment,'evaluation.lock')):
            lock=(run/name).open('a');locks.append(lock);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=read(source/'plan.json');check_plan(original);check_plan(read(assessment/'plan.json'))
        if original['curriculum'] != dict(stage='quarter',fraction=.5):
            raise ValueError('Expected completed mixed-start control')
        if original.get('relative_potential_coefficient',0.) != 0.:
            raise ValueError('Control already has restored relative shaping')
        controls=[read(f) for f in (source/'evaluations').glob('*/completion.json')]
        tested=[read(f) for f in (assessment/'evaluations').glob('*/completion.json')]
        broad_base=next(e for e in controls if e['baseline'] and e['kind']=='development')
        candidate_broads=[e for e in controls+tested if not e['baseline'] and e['kind']=='development']
        if len(candidate_broads)!=3 or any(e['summary']['wins']>broad_base['summary']['wins'] for e in candidate_broads):
            raise ValueError('Review the improved or incomplete candidate set before reverting to baseline')
        if any(read(source/f's{s}_budget_decision.json')['extend'] for s in original['seeds']):
            raise ValueError('Unexpected extension; review source')
        checkpoint=Path(read(source/'baseline_checkpoint.json')['path']);meta=verify(checkpoint)
        if not meta['baseline']:
            raise ValueError('Expected preserved baseline checkpoint')
        origin=Path(original['source_run']);check_plan(read(origin/'plan.json'))
        agreement=read(origin/'baseline_native_agreement.json')
        if not agreement['exact'] or agreement['games']!=128:
            raise ValueError('No exact native baseline lineage')
        panels=[]
        for kind in ('monitor','development'):
            folders=[f.parent for f in (source/'evaluations').glob('*/completion.json')
                     if read(f)['baseline'] and read(f)['kind']==kind]
            if len(folders)!=1:raise ValueError('Ambiguous source baseline')
            entry,rows,files=validate_cached_panel(original,folders[0])
            if entry['checkpoint_id']!=meta['checkpoint_id']:raise ValueError('Baseline identity differs')
            if kind=='monitor' and verify(Path(entry['reused_from_evaluation'])/'input')!=meta:
                raise ValueError('Inherited evaluated baseline differs')
            panels.append((folders[0],entry,rows,files))
        seeds=[45500,45501];bands=[2480400000,2490400000]
        for file in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            r=read(file)
            if any(b<r['stop_exclusive'] and b+1000000>r['start'] for b in bands):
                raise ValueError('Training band reserved: '+str(file))
        torch.set_num_threads(1)
        model=PPO.load(checkpoint/'learner.zip',device='cpu')
        expected=read(source/'parent_verification.json')['state']
        state=dict(policy=tensor_digest(model.policy.state_dict()),optimizer=tensor_digest(model.policy.optimizer.state_dict()),
                   decisions=model.num_timesteps,updates=model._n_updates)
        if state!=expected or model.observation_space.shape!=(43,):
            raise ValueError('Preserved actor/critic/Adam/counters differ')
        out.mkdir(parents=True);prepared=out/'selected_parent.zip';shutil.copyfile(checkpoint/'learner.zip',prepared)
        if sha(prepared)!=sha(checkpoint/'learner.zip'):raise ValueError('Parent copy changed')
        text=(source/'aim_cpu_runner.py').read_text();old='from experiments.aim_async.mixed_start_worker import make_worker'
        if text.count(old)!=1:raise ValueError('Worker integration boundary changed')
        runner=out/'aim_cpu_runner.py';runner.write_text(text.replace(old,'from experiments.aim_async.relative_reward_worker import make_worker'))
        plan=copy.deepcopy(original)
        plan.update(parent=str(prepared.relative_to(ROOT)),parent_source=str((checkpoint/'learner.zip').relative_to(ROOT)),
                    selected_parent_checkpoint=str(checkpoint),source_run=str(source),source_assessment=str(assessment),
                    seeds=seeds,training_bands=bands,relative_potential_coefficient=1.,
                    intervention=dict(factor='general relative-state potential coefficient',control=0.,value=1.,
                                      unchanged='preserved43-feature baseline/Adam/PPO, aiming signal, damage exchange, quarter resets,48foes and fixed evaluation'),
                    comparison_scope='Adaptive two-seed reward comparison. Fresh RNGs; consumed development, not unseen final evidence.')
        inputs=[prepared,runner,Path(__file__).resolve(),ROOT/'experiments/aim_async/relative_reward_worker.py',
                source/'plan.json',source/'parent_verification.json',origin/'baseline_native_agreement.json',checkpoint/'READY.json',
                assessment/'ASSESSMENT_INPUTS.json',assessment/'evaluation_complete.json']
        inputs += [f for f in (assessment/'evaluations').glob('*/completion.json')]
        for folder,entry,rows,files in panels:inputs += [folder/'completion.json',*files]
        plan['frozen_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in inputs})
        plan['config_id']=identity({k:v for k,v in plan.items() if k not in ('config_id','frozen_sha256')})
        write(out/'plan.json',plan)
        for folder,entry,rows,files in panels:
            key=identity(dict(source=str(folder),config_id=plan['config_id']))
            reuse=reused_baseline(entry,folder,source,seeds[0])
            reuse['reuse_basis']='Byte-identical evaluated43-feature baseline and fullAdam/counters; inherited exact native baseline agreement. Only training general shaping changes; same neutral evaluation and800paired cells.'
            write(out/'evaluations'/key/'completion.json',reuse);write(out/'candidates'/f'{key}.json',reuse)
            for row in rows:
                cell_id=identity({k:row[k] for k in ('opponent','seed','seat','action_seed')})
                write(out/'evaluations'/key/'cells'/f'{cell_id}.json',row)
        write(out/'baseline_checkpoint.json',dict(path=str(checkpoint)))
        write(out/'parent_verification.json',dict(exact=True,state=state,source_sha256=sha(checkpoint/'learner.zip'),
              prepared_sha256=sha(prepared),reused_baseline_games=800,new_baseline_games=0))
        for band in bands:
            with (ROOT/f'runs/training_reservation_{band}.json').open('x') as f:
                json.dump(dict(run=str(out),start=band,stop_exclusive=band+1000000),f)
        write(out/'started.json',dict(time=time.time(),status='prepared_not_launched'))
        write(out/'goal_control.json',dict(status='prepared',automatic_learning=True,publication=False,
              effective_objective=plan['objective'],next_action='One read-only review, then launch two fixed-reward repetitions.'))
        check_plan(original);check_plan(plan)
        print(json.dumps(dict(out=str(out),prepared=True,reused_baseline_games=800,new_baseline_games=0,intervention=plan['intervention'])))
    finally:
        for lock in locks:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('source','assessment','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();prepare(args.source.resolve(),args.assessment.resolve(),args.out.resolve())
