"""Prepare an isolated predictive-aim continuation and compressed fixed panels."""
import argparse
import copy
import fcntl
import json
import runpy
import shutil
import time
from pathlib import Path
from experiments.rear_async.storage import identity,read,sha,verify,write
from experiments.aim_async.roster import fingerprints,select_representatives
from experiments.aim_async.evaluate import profile
from tools.prepare_rear_cpu_selected_parent import validate_panel,reused_baseline
from tools.prepare_rear_cpu_entropy import tensor_digest
from tools.rear_async_cpu import ROOT,check_plan

SOURCE=ROOT/'runs/rear_cpu_joint585_lr3e5_20261013'
ASSESSMENT=ROOT/'runs/rear_cpu_joint585_lr3e5_early393_20261013'
CHECKPOINT=SOURCE/'checkpoints/s45200_step_003932160'
EXPECTED_SHA='12cf61746e402031451ea995e39f58fc517b263c134d25c81a5a1f3d0637efae'


def runner_source():
    source=(ROOT/'tools/rear_async_cpu.py').read_text()
    replacements={
        'ROOT=Path(__file__).resolve().parents[1]':'from tools.rear_async_cpu import ROOT',
        'from tools.league_residual_export import export':'from tools.aim_cpu_export import export',
        'from experiments.rear_async.worker import make_worker':'from experiments.aim_async.worker import make_worker',
        'from experiments.rear_async.evaluate import cell,profile':'from experiments.aim_async.evaluate import cell,profile',
        'from experiments.rear_async.judgement import budget_decision':'from experiments.aim_async.judgement import budget_decision',
        'from experiments.rear_async.judgement import candidate_status':'from experiments.aim_async.judgement import candidate_status',
        "[sys.executable,'-m','tools.rear_async_cpu','evaluate'":"[sys.executable,str(Path(__file__).resolve()),'evaluate'",
        "decision=budget_decision(points,base,added,plan['minimum_steps'],plan['maximum_steps'])":
        "broad_summary=next(p['summary'] for p in completed if p.get('seed')==seed and p['additional_steps']==added and p['kind']=='development')\n                        broad_base=next(p['summary'] for p in completed if p['baseline'] and p['kind']=='development')\n                        decision=budget_decision(points,base,added,plan['minimum_steps'],plan['maximum_steps'],broad=broad_summary,broad_baseline=broad_base)",
        'Bounded rear-metric budget decisions recorded;all checkpoints preserved':'Bounded combat/aim budget decisions recorded;all checkpoints preserved',
        "restart=__import__('torch').load(latest/'workers.pt',weights_only=False) if latest else None":
        "if not (out/f's{seed}_started.json').exists():write(out/f's{seed}_started.json',dict(time=time.time()))\n            restart=__import__('torch').load(latest/'workers.pt',weights_only=False) if latest else None",
        'finally:vec.close()':"finally:vec.close()\n            write(out/f's{seed}_learning_complete.json',dict(time=time.time()))",

        "if failure['attempts']>=3:write(folder/'blocked.json',failure)":"write(folder/'blocked.json',failure)  # No blind retries; diagnose before resuming.",
        "if (out/'STOP').exists():raise InterruptedError('User STOP marker')":
        "if (out/'STOP').exists():raise InterruptedError('User STOP marker')\n            if self.num_timesteps%2048==0 and list((out/'evaluations').glob('*/blocked.json')):raise RuntimeError('Evaluation blocked; diagnose before resuming')",
        'verify(copy);summary=profile(rows)':
        "verify(copy);summary=profile(rows)\n    if m['baseline'] and job['kind']=='monitor':\n        from tools.prepare_aim_cpu import verify_baseline_cells\n        verify_baseline_cells(out,rows)",
    }
    for old,new in replacements.items():
        if source.count(old)!=1:raise ValueError('Runner integration boundary changed: '+old)
        source=source.replace(old,new)
    old="            ready=[p for p in (out/'checkpoints').glob(f's{seed}_step_*') if (p/'READY.json').exists()]\n            latest=max(ready,key=lambda p:verify(p)['additional_steps']) if ready and resume else None"
    new="            from experiments.aim_async.runtime import latest_training_checkpoint\n            latest=latest_training_checkpoint(out,seed) if resume else None"
    if source.count(old)!=1:raise ValueError('Resume checkpoint boundary changed')
    source=source.replace(old,new)
    old="    else:check_plan(read(out/'plan.json'))"
    new=old+"\n    launch_lock=(out/'launch.lock').open('a');fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n    from experiments.aim_async.runtime import prepare_resume\n    prepare_resume(out)"
    if source.count(old)!=1:raise ValueError('Launch lifecycle boundary changed')
    source=source.replace(old,new)
    old="    report(out);write(out/'evaluation_complete.json',dict(status='stopped' if (out/'STOP').exists() else 'queue_drained'))"
    new="    from experiments.aim_async.runtime import evaluation_status\n    write(out/'evaluation_complete.json',dict(status=evaluation_status(out)));report(out)"
    if source.count(old)!=1:raise ValueError('Evaluator completion boundary changed')
    source=source.replace(old,new)
    start=source.index('def report(out):');end=source.index('\ndef launch(',start)
    source=source[:start]+"def report(out):\n    from tools.aim_cpu_reporting import update\n    update(out)\n"+source[end:]
    return source


def verify_baseline_cells(out,rows):
    old=read(out/'baseline_monitor_expected.json')
    key=lambda r:(r['opponent'],r['seed'],r['seat'],r['action_seed'])
    indexed={key(r):r for r in old}
    if {key(r) for r in rows}!=set(indexed):raise ValueError('Baseline monitor identity changed')
    fields=('result','own_health','opp_health','steps','duration','damage_dealt','damage_received','rear_damage')
    differences=[dict(key=key(r),field=f,old=indexed[key(r)][f],new=r[f]) for r in rows for f in fields if r[f]!=indexed[key(r)][f]]
    write(out/'baseline_native_agreement.json',dict(games=len(rows),exact=not differences,differences=differences))
    if differences:raise ValueError('Migrated baseline disagrees with archived native outcomes; cached broad reuse must be reviewed')


def contracted(value):
    import torch
    from experiments.aim_async.policy import original_columns
    if isinstance(value,torch.Tensor):
        if value.ndim==2 and value.shape[1]==43:
            if torch.count_nonzero(value[:,31:34]):raise ValueError('Added weights/moments must start at zero')
            return value[:,original_columns()]
        return value
    if isinstance(value,dict):return {k:contracted(v) for k,v in value.items()}
    if isinstance(value,list):return [contracted(v) for v in value]
    if isinstance(value,tuple):return tuple(contracted(v) for v in value)
    return value


def behavior_check(old,model,plan,baseline):
    """Actual archive actions + short native worker integration; no full new games."""
    import numpy as np
    import torch
    from tools.league_matches import Actor
    from experiments.aim_async.worker import make_worker
    from experiments.league.residual_ppo_policy import actor_parameters
    from experiments.league.residual_features import numpy_logits as old_logits
    from experiments.aim_async.features import numpy_logits
    from experiments.aim_async.policy import original_columns
    from aircombat_gym.wvr import obs as O
    rng=np.random.default_rng(739)
    x=rng.uniform(-1,1,(64,40)).astype(np.float32)
    extra=np.zeros((64,43),dtype=np.float32);extra[:,original_columns()]=x;extra[:,31:34]=rng.uniform(-1,1,(64,3))
    with torch.no_grad():
        if not torch.equal(old.policy.predict_values(torch.tensor(x)),model.policy.predict_values(torch.tensor(extra))):raise ValueError('Real critic changed')
        if not torch.equal(old.policy.get_distribution(torch.tensor(x)).distribution.logits,model.policy.get_distribution(torch.tensor(extra)).distribution.logits):raise ValueError('Real actor changed')
    np.testing.assert_array_equal(old_logits(x,actor_parameters(old.policy)),numpy_logits(extra,actor_parameters(model.policy)))
    count=0
    for action_seed in ('4900','4901'):
        old_actor=Actor(verify(CHECKPOINT)['sampled_specs'][action_seed]);new_actor=Actor(verify(baseline)['sampled_specs'][action_seed])
        d={name:i for i,name in enumerate(O.STATE_NAMES)}
        for episode in range(4):
            old_actor.reset();new_actor.reset()
            for step in range(64):
                raw=np.zeros(39,dtype=np.float32)
                for side in ('own','opp'):
                    for name in ('x','y'):raw[d[side+'_'+name]]=rng.uniform(-5000,5000)
                    raw[d[side+'_h']]=5000
                    raw[d[side+'_psi']]=rng.uniform(-np.pi,np.pi)
                    raw[d[side+'_theta']]=rng.uniform(-.1,.1)
                    for name in ('vx','vy'):raw[d[side+'_'+name]]=rng.uniform(-250,250)
                    raw[d[side+'_health']]=1.
                    raw[d[side+'_dist_to_boundary']]=50000
                raw[d['t_remaining']]=120-step*.05
                if old_actor.act(obs=raw)!=new_actor.act(obs=raw):raise ValueError('Sampled exported action changed')
                count+=1
    worker=make_worker(dict(plan,seed=plan['seeds'][0],ic_band=plan['training_bands'][0]),0)
    try:
        state,info=worker.reset(seed=plan['seeds'][0])
        for _ in range(2):
            action,_=model.predict(state,deterministic=True)
            state,reward,term,trunc,info=worker.step(action)
            if state.shape!=(43,) or not np.isfinite(state).all() or not np.isfinite(reward):raise ValueError('Invalid native worker output')
    finally:worker.close()
    return dict(real_actor_critic_logits_exact=True,exported_sampled_actions_equal=count,native_worker_steps=2,completed_simulation_games=0)


def prepare(out):
    import psutil
    import torch
    from stable_baselines3 import PPO
    from experiments.aim_async.policy import expand_learner
    if out.exists():raise FileExistsError('Use a fresh output directory')
    if not out.is_relative_to(ROOT/'runs'):raise ValueError('Output outside runs')
    for run in (SOURCE,ASSESSMENT):
        if not (run/'training_complete.json').exists() or (run/'training_failure.json').exists():raise ValueError('Source not terminal')
        if read(run/'evaluation_complete.json')['status']!='queue_drained':raise ValueError('Source queue not drained')
    for process in psutil.process_iter(['pid','cmdline','status']):
        cmd=' '.join(process.info['cmdline'] or [])
        if process.info['status']!=psutil.STATUS_ZOMBIE and ('tools.rear_async_cpu launch' in cmd or 'aim_cpu_runner.py launch' in cmd):
            raise RuntimeError('Existing trainer found: '+str(process.pid))
    if psutil.virtual_memory().available<10*2**30:raise MemoryError('Need10GiB available')
    locks=[]
    try:
        for run in (SOURCE,ASSESSMENT):
            for name in ('training.lock','evaluation.lock'):
                lock=(run/name).open('a');locks.append(lock);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source_plan=read(SOURCE/'plan.json');check_plan(source_plan);check_plan(read(ASSESSMENT/'plan.json'))
        meta=verify(CHECKPOINT)
        if sha(CHECKPOINT/'learner.zip')!=EXPECTED_SHA:raise ValueError('Unexpected selected source')
        panels={}
        for kind,run in (('monitor',SOURCE),('development',ASSESSMENT)):
            folders=[p.parent for p in (run/'evaluations').glob('*/completion.json') if read(p)['kind']==kind and read(p)['checkpoint_id']==meta['checkpoint_id']]
            if len(folders)!=1:raise ValueError('Ambiguous selected-parent panel')
            folder=folders[0];panels[kind]=(folder,validate_panel(source_plan,CHECKPOINT,folder))
        seeds=[45300,45301];bands=[2440400000,2450400000]
        for file in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            r=read(file)
            if any(b<r['stop_exclusive'] and b+1000000>r['start'] for b in bands):raise ValueError('Training band reserved: '+str(file))
        plan=copy.deepcopy(source_plan)
        rosters={};evidence_paths=[]
        for kind in ('monitor','development'):
            folders=[p.parent for p in (SOURCE/'evaluations').glob('*/completion.json')
                if read(p)['kind']==kind and not read(p)['baseline'] and not read(p).get('reused_from_run')]
            if panels[kind][0] not in folders:folders.append(panels[kind][0])
            evidence=fingerprints(folders)
            selected,report=select_representatives(plan[kind+'_opponents'],evidence)
            report['source_folders']=[str(p) for p in folders]
            report['source_completion_sha256']={str(p/'completion.json'):sha(p/'completion.json') for p in folders}
            rosters[kind]=report;plan[kind+'_opponents']=selected
            evidence_paths += [p/'completion.json' for p in folders]
        torch.set_num_threads(1);old=PPO.load(CHECKPOINT/'learner.zip',device='cpu')
        model=expand_learner(old);out.mkdir(parents=True)
        prepared=out/'selected_parent.zip';model.save(prepared);model=PPO.load(prepared,device='cpu')
        before=dict(policy=tensor_digest(old.policy.state_dict()),optimizer=tensor_digest(old.policy.optimizer.state_dict()),
            decisions=old.num_timesteps,updates=old._n_updates)
        after=dict(policy=tensor_digest(contracted(model.policy.state_dict())),optimizer=tensor_digest(contracted(model.policy.optimizer.state_dict())),
            decisions=model.num_timesteps,updates=model._n_updates)
        if before!=after:raise ValueError('Actual actor/critic/Adam/counters not preserved')
        plan.update(parent=str(prepared.relative_to(ROOT)),parent_source=str((CHECKPOINT/'learner.zip').relative_to(ROOT)),
            selected_parent_checkpoint=str(CHECKPOINT),source_run=str(SOURCE),seeds=seeds,training_bands=bands,
            aim_coefficients=dict(potential=.05,damage=.5,received=.5),
            tail_coefficients=dict(potential=0.,hold=0.,damage=0.,received=0.),position_per_second=0.,rear_damage_coefficient=0.,
            curriculum=dict(stage='disabled',fraction=0.),
            objective='Predictive aiming, actual damage exchange and native CPU win rate. Rear development held by latest user.',
            candidate_rule='Preserve every checkpoint and small gain. Keep one empirically distinct evaluation representative per historical cluster; retain archived members.',
            intervention=dict(features='40 to43 analytic public lead cues with zero initialization',reward='terminal health utility +0.5dealt -0.5received +0.05potential difference',curriculum='original FairFight IC sampling; no rear-biased starts'),
            comparison_scope='Adaptive consumed development, changed reward/features/initial sampling, fresh RNGs. Not a causal single-factor trial or unseen-final evidence.',
            baseline_monitor_expected=str(out/'baseline_monitor_expected.json'),evaluation_roster_report=str(out/'evaluation_roster.json'))
        for key in ('maximum_update_epochs_per_checkpoint','maximum_update_epochs_per_minimum','maximum_update_epochs_per_cap'):
            if key in plan:plan[key]=int(plan[key])
        runner=out/'aim_cpu_runner.py';runner.write_text(runner_source())
        write(out/'evaluation_roster.json',rosters)
        monitor_ids={f['id'] for f in plan['monitor_opponents']}
        old_monitor=[read(p) for p in (panels['monitor'][0]/'cells').glob('*.json') if read(p)['opponent'] in monitor_ids]
        write(out/'baseline_monitor_expected.json',old_monitor)
        inputs=[prepared,runner,Path(__file__).resolve(),ROOT/'tools/aim_cpu_reporting.py',ROOT/'tools/aim_cpu_export.py',
            out/'evaluation_roster.json',out/'baseline_monitor_expected.json',CHECKPOINT/'learner.zip',CHECKPOINT/'READY.json',
            *list((ROOT/'experiments/aim_async').glob('*.py')),*evidence_paths]
        plan['frozen_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in inputs})
        plan['config_id']=identity({k:v for k,v in plan.items() if k not in ('config_id','frozen_sha256')})
        write(out/'plan.json',plan);write(out/'started.json',dict(time=time.time(),status='prepared_not_launched'))
        runner_functions=runpy.run_path(str(runner),run_name='aim_preparation')
        baseline=runner_functions['checkpoint'](model,None,out,seeds[0],model.num_timesteps,0,0,0,baseline=True)
        behavior=behavior_check(old,model,plan,baseline)
        folder,entry=panels['development'];ids={f['id'] for f in plan['development_opponents']}
        rows=[read(p) for p in (folder/'cells').glob('*.json') if read(p)['opponent'] in ids]
        reuse=reused_baseline(entry,folder,ASSESSMENT,seeds[0])
        reuse.update(checkpoint_id=baseline.name,checkpoint=str(baseline),source_checkpoint=str(CHECKPOINT),
            summary=profile(rows),per_opponent={k:profile([r for r in rows if r['opponent']==k]) for k in sorted(ids)},
            reused_cells_in_final_attempt=len(rows),
            reuse_basis='Source actor/critic/Adam restored exactly on original columns; new weights zero; exact sampled export action probes. Native monitor replay must match archived HP/steps/verdicts. Same ICs, seats and action seeds; compressed roster subset reaggregated.')
        key=identity(dict(reused_subset=str(folder),config_id=plan['config_id']))
        write(out/'evaluations'/key/'completion.json',reuse);write(out/'candidates'/f'{key}.json',reuse)
        for row in rows:
            cell_key=identity(dict(opponent=row['opponent'],seed=row['seed'],seat=row['seat'],action_seed=row['action_seed']))
            write(out/'evaluations'/key/'cells'/f'{cell_key}.json',row)
        write(out/'baseline_checkpoint.json',dict(path=str(baseline)))
        runner_functions['request'](out,baseline,'monitor')
        shutil.copyfile(ASSESSMENT/'development_candidate_comparison.json',out/'inherited_candidate_comparison.json')
        for band in bands:
            with (ROOT/f'runs/training_reservation_{band}.json').open('x') as f:json.dump(dict(run=str(out),start=band,stop_exclusive=band+1000000),f)
        write(out/'parent_verification.json',dict(before=before,contracted_after=after,exact_original_parameters=True,
            source_sha256=EXPECTED_SHA,prepared_sha256=sha(prepared),behavior=behavior,
            new_baseline_games_planned=len(old_monitor),reused_baseline_games=len(rows),
            original_full_parent_development=entry['summary'],compressed_parent_development=reuse['summary']))
        write(out/'goal_control.json',dict(status='prepared',effective_objective=plan['objective'],
            automatic_learning=True,publication=False,small_gains_retained=True,similar_foes_use_one_representative=True,
            next_action='Finish review; launch one native CPU trainer/evaluator pair. Repeat automatically under active Goal after bounded cycle review.'))
        check_plan(plan)
        print(json.dumps(dict(status='prepared',out=str(out),parent_decisions=model.num_timesteps,
            roster_counts={k:(v['original_opponents'],v['selected_opponents']) for k,v in rosters.items()},
            parent_verification=behavior,new_baseline_games=len(old_monitor),reused_baseline_games=len(rows))))
    finally:
        for lock in locks:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.out.resolve())
