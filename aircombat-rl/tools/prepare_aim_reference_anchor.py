"""Prepare fixed-reference PPO continuation from preserved native500 state."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import shutil
import time
import zipfile
from pathlib import Path
from experiments.rear_async.storage import identity,read,sha,verify,write
from tools.rear_async_cpu import ROOT,check_plan


def choose_full(entries):
    full=[e for e in entries if e['kind']=='development']
    if any(e['summary']['games']!=672 for e in full):
        raise ValueError('Incomplete full native panel')
    baselines=[e for e in full if e['baseline']]
    candidates=[e for e in full if not e['baseline']]
    if len(baselines)!=1 or not candidates:
        raise ValueError('No qualified full-panel improvement')
    chosen=max(candidates,key=lambda e:(e['summary']['wins'],
        e['summary'].get('damage_dealt',0)-e['summary'].get('damage_received',0),
        -e.get('additional_steps',0)))
    if chosen['summary']['wins']<=baselines[0]['summary']['wins']:
        raise ValueError('No qualified full-panel improvement')
    return chosen


def short_budget():
    return dict(minimum_steps=3932160,maximum_steps=5242880,checkpoint_steps=655360,monitor_steps=1310720,maximum_update_epochs_per_minimum=768,maximum_update_epochs_per_cap=1024,maximum_update_epochs_per_checkpoint=128)


def prepare(source,out):
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a fresh output under runs')
    if not (source/'training_complete.json').exists() or (source/'training_failure.json').exists():
        raise RuntimeError('Source not successfully terminal')
    if read(source/'evaluation_complete.json')['status']!='queue_drained':
        raise RuntimeError('Source evaluation not drained')
    for path in (source/'queue').glob('*.json'):
        folder=source/'evaluations'/read(path)['job_id']
        if not (folder/'completion.json').exists() or (folder/'blocked.json').exists():
            raise RuntimeError('Source queue incomplete or blocked')
    import psutil
    for proc in psutil.process_iter(['cmdline','status']):
        args=proc.info['cmdline'] or []
        if proc.info['status']!=psutil.STATUS_ZOMBIE and any(Path(a).name=='aim_cpu_runner.py' or a=='tools.rear_async_cpu' for a in args) and any(a in args for a in ('launch','evaluate')):
            raise RuntimeError('Existing native trainer/evaluator:'+str(proc.pid))
    if psutil.virtual_memory().available<10*2**30:raise MemoryError('Need10GiB available')
    locks=[]
    try:
        for name in ('launch.lock','training.lock','evaluation.lock'):
            lock=(source/name).open('a');locks.append(lock);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=read(source/'plan.json');check_plan(original)
        diagnosis=ROOT/'runs/aim_cpu_mode_probe_20261014'
        if read(diagnosis/'ANALYSIS.json')['status']!='completed_all_diagnostic_cells_analyzed':
            raise ValueError('Execution-mode diagnosis not analyzed')
        check_plan(read(diagnosis/'plan.json'))
        review=read(source/'TWO_REPEAT_REVIEW.json');audit=read(source/'CYCLE_LOG_AUDIT.json')
        if review['status']!='two_repeats_complete_analyzed' or set(audit['training'])!={str(s) for s in original['seeds']}:
            raise ValueError('Source repeats not completely analyzed')
        for seed in original['seeds']:
            trained=audit['training'][str(seed)]
            if trained['all_episodes']['games']!=read(source/f's{seed}_result.json')['training_games'] or trained['nonfinite_ppo_fields']:
                raise ValueError('Completed audit differs or nonfinite')
        if (original['relative_enemy_feedback']!='public_enemy_los_cos_v1' or original['ppo']['learning_rate']!=1e-5
                or original['terminal_draw_penalty']!=.25 or original['learner_feature_dim']!=43 or original['prior_bias']!=6.
                or original['processes']!=8 or original['workers']!=32
                or original['aim_coefficients']!=dict(potential=.05,damage=.5,received=.75) or original['ppo']['ent_coef']!=.003):
            raise ValueError('Expected completed originalLOS/received0.75/rate1e-5/draw0.25 control')
        entries=[(p,read(p)) for p in (source/'evaluations').glob('*/completion.json')]
        chosen=choose_full([e for _,e in entries])
        checkpoint=Path(chosen['checkpoint']).resolve();metadata=verify(checkpoint)
        if str(checkpoint)!=review['selected_peak']['path'] or chosen['summary']['wins']!=review['selected_peak']['wins'] or chosen['checkpoint_id']!=metadata['checkpoint_id']:
            raise ValueError('Reviewed full-panel parent identity differs')
        parent=checkpoint/'learner.zip'
        with zipfile.ZipFile(parent) as archive:
            serialized=json.loads(archive.read('data'))
            before=dict(policy=hashlib.sha256(archive.read('policy.pth')).hexdigest(),
                optimizer=hashlib.sha256(archive.read('policy.optimizer.pth')).hexdigest(),
                decisions=serialized['num_timesteps'],updates=serialized['_n_updates'])
        for key in ('learning_rate','n_steps','n_epochs','batch_size','gamma','gae_lambda','target_kl','ent_coef','max_grad_norm'):
            if serialized[key]!=original['ppo'][key]:raise ValueError('Saved parent PPO differs:'+key)
        exported=read(checkpoint/'greedy/policy_net.json')
        if serialized['num_timesteps']!=metadata['total_steps'] or serialized['n_envs']!=32 or serialized['policy_kwargs']['prior_bias']!=6. or exported['observation_dim']!=43 or exported['prior_bias']!=6. or exported['teacher']!=original['teacher']:
            raise ValueError('Saved parent counters/feature/teacher contract differs')
        from experiments.aim_async.evaluate import profile
        panels=[]
        for kind in ('monitor','development'):
            matches=[(p,e) for p,e in entries if e['checkpoint_id']==metadata['checkpoint_id'] and not e['baseline'] and e['kind']==kind]
            if len(matches)!=1:raise ValueError('Ambiguous qualified panel')
            path,entry=matches[0];files=sorted((path.parent/'cells').glob('*.json'));rows=[read(f) for f in files]
            samplers=original['monitor_action_seeds'] if kind=='monitor' else [4900,4901]
            expected={(f['id'],original[kind+'_band']+i,seat,action) for f in original[kind+'_opponents'] for i in range(original[kind+'_ics']) for seat in ('red','blue') for action in samplers}
            keys={tuple(r[k] for k in ('opponent','seed','seat','action_seed')) for r in rows}
            if len(rows)!=len(expected) or keys!=expected:raise ValueError('Incomplete or unpaired qualified panel')
            measured=profile(rows)
            if measured.keys()!=entry['summary'].keys() or any(measured[k]!=v and not (measured[k] is not None and v is not None and math.isclose(measured[k],v,rel_tol=0.,abs_tol=1e-12)) for k,v in entry['summary'].items()):
                raise ValueError('Qualified native profile differs')
            panels.append((path,entry,rows,files))
        full=next(rows for _,e,rows,_ in panels if e['kind']=='development');monitor=next(rows for _,e,rows,_ in panels if e['kind']=='monitor')
        indexed=lambda rows:{tuple(r[k] for k in ('opponent','seed','seat','action_seed')):r for r in rows}
        if len(full)!=672 or len(monitor)!=128 or any(indexed(full).get(k)!=r for k,r in indexed(monitor).items()):
            raise ValueError('Monitor not exact qualified full subset')
        seeds,bands=[47060,47061],[2940400000,2950400000]
        for path in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            reservation=read(path)
            if any(b<reservation['stop_exclusive'] and b+1000000>reservation['start'] for b in bands):raise ValueError('Reserved training band:'+str(path))
        out.mkdir(parents=True);prepared=out/'qualified500_parent.zip';shutil.copyfile(parent,prepared)
        if sha(prepared)!=sha(parent):raise ValueError('Copied qualified learner bytes differ')
        runner=out/'aim_cpu_runner.py';code=(source/'aim_cpu_runner.py').read_text();old='from tools.aim_cpu_received_weight_reporting import update'
        if code.count(old)!=1:raise ValueError('Reporter boundary differs')
        code=code.replace(old,'from tools.aim_cpu_reference_reporting import update')
        loader='from stable_baselines3 import PPO'
        if code.count(loader)!=1:raise ValueError('PPO loader boundary differs')
        code=code.replace(loader,'from experiments.aim_async.reference_ppo import ReferenceAnchoredPPO as PPO')
        line="                model=PPO.load(latest/'learner.zip' if latest else ROOT/plan['parent'],env=vec,device='cpu',seed=seed,force_reset=True,n_steps=640)"
        if code.count(line)!=1:raise ValueError('PPO load call boundary differs')
        code=code.replace(line,line+"\n                model.attach_reference(ROOT/plan['reference_archive'],plan['reference_kl_coefficient'])\n                assert model.reference_policy_hash==plan['reference_policy_sha256']")
        compile(code,str(runner),'exec');runner.write_text(code)
        budget=out/'SHORT_CONTINUATION_BUDGET.json'
        settings=short_budget()
        write(budget,dict(settings,reason='Preserve existing three-monitor/live/resume gate andunanchoredshort500controlbudget;fixedreferenceKL penalty is theonlytrainingintervention.'))
        expected_monitor=out/'baseline_monitor_expected.json';write(expected_monitor,monitor)
        plan=copy.deepcopy(original);plan.update(reference_archive=str(prepared.relative_to(ROOT)),reference_kl_coefficient=.1,reference_policy_sha256=before['policy'],analysis_run=str(diagnosis),comparison_control_run=str(ROOT/'runs/aim_cpu_short500_resume_20261014'),parent=str(prepared.relative_to(ROOT)),parent_source=str(parent),source_run=str(source),
            selected_parent_checkpoint=str(checkpoint),selected_parent_assessment=str(source/'S47021_MINIMUM_FULL_REVIEW.json'),
            seeds=seeds,training_bands=bands,minimum_steps=settings['minimum_steps'],maximum_steps=settings['maximum_steps'],
            maximum_update_epochs_per_minimum=settings['maximum_update_epochs_per_minimum'],maximum_update_epochs_per_cap=settings['maximum_update_epochs_per_cap'],
            budget_spec_repository_relative=str(budget.relative_to(ROOT.parent)),baseline_monitor_expected=str(expected_monitor),
            scope='Fixed-reference KL CPU PPO continuation from provisional highest native500/672checkpoint.',
            intervention=dict(factor='actor_reference_kl_penalty',control=0.,value=.1,
                parent='Nativequalified500checkpoint;provisionaladaptive selection,not128monitor peak.',
                unchanged='NN/initializedAdam/counters/initiallearnerbytes/43features/teacher/prior6/allrewards/standardPPOsettings/48foes/probabilities/quarter0.5/nativephysics/guns/verdict/evaluation/shortbudget'),
            comparison_scope='FixedreferenceKL0.1 versusunanchoredshort500control483/478;differenttrainingRNG/bands,exploratory notcausalconfirmation. Consumed672development is not unseen finalvalidation.',
            parent_choice='Highest complete native672candidate500;preserve499/498fallbacks;onewin is not statisticalconfirmation.',
            qualified_parent_full_wins=500,qualified_parent_full_games=672,
            historical_reference_fields='Inherited legacy reference_* fields record earlier leaguehistory;activecomparator is cached500/672selected_parent_checkpoint.')
        inputs=[prepared,runner,budget,expected_monitor,Path(__file__).resolve(),ROOT/'tests/test_aim_reference_ppo.py',ROOT/'tools/aim_cpu_reference_reporting.py',ROOT/'experiments/aim_async/reference_ppo.py',checkpoint/'READY.json']
        inputs.extend([diagnosis/name for name in ('plan.json','completion.json','ANALYSIS.json','REPORT.md','NEXT_REFERENCE_ANCHOR_PLAN.md','REVIEW.json')])
        inputs.extend([ROOT/'runs/aim_cpu_short500_resume_20261014/TWO_REPEAT_REVIEW.json'])
        import inspect
        from stable_baselines3 import PPO
        from stable_baselines3.common.base_class import BaseAlgorithm
        from stable_baselines3.common.save_util import load_from_zip_file
        inputs.extend([Path(inspect.getfile(obj)) for obj in (PPO,BaseAlgorithm,load_from_zip_file)])
        inputs.extend(checkpoint/name for name in metadata['files'])
        inputs.extend(source/name for name in ('plan.json','parent_verification.json','CYCLE_LOG_AUDIT.json','TWO_REPEAT_REVIEW.json',
            'S47020_COMPLETED_REVIEW.json','S47021_COMPLETED_REVIEW.json','S47021_MINIMUM_FULL_REVIEW.json','REPEAT_RESULTS.md',
            'NEXT_SHORT_CONTINUATION_PLAN.md','REVIEW.json','training_complete.json','evaluation_complete.json'))
        inputs.extend(source/name for name in audit['read_files'])
        for path,_,_,files in panels:inputs.extend([path,*files])
        plan['frozen_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in inputs})
        plan['config_id']=identity({k:v for k,v in plan.items() if k not in ('config_id','frozen_sha256')});write(out/'plan.json',plan)
        for path,entry,rows,_ in panels:
            folder=out/'evaluations'/identity(dict(config_id=plan['config_id'],parent=metadata['checkpoint_id'],kind=entry['kind']))
            reused=dict(entry,seed=seeds[0],additional_steps=0,baseline=True,status='reused_byte_identical_qualified500_initial_learner',actual_new_games=0,
                new_games_in_final_attempt=0,reused_cells_in_final_attempt=len(rows),reused_from_sibling_games=0,wall_seconds=0.,
                reused_from_run=str(source),reused_from_evaluation=str(path.parent),
                reuse_basis='Qualified500learnerZIP byte-identical;NN/initializedAdam/counters/features/teacher/prior/nativeinference unchanged. Budget/RNG only affect futurelearning.')
            write(folder/'completion.json',reused);write(out/'candidates'/f'{folder.name}.json',reused)
            for row in rows:write(folder/'cells'/f"{identity({k:row[k] for k in ('opponent','seed','seat','action_seed')})}.json",row)
        write(out/'baseline_checkpoint.json',dict(path=str(checkpoint)))
        write(out/'parent_verification.json',dict(before=before,after=before,restored=before,state_exact=True,source_sha256=sha(parent),prepared_sha256=sha(prepared),
            byte_identical=True,byte_identical_to_qualified_parent=True,state_basis='Byte-identical serializedlearnercopy,not an additional model-load test.',
            policy_behavior_unchanged=True,initial_native_behavior_preserved=True,comparator_wins=500,comparator_games=672,
            baseline_unique_games=672,baseline_panel_entries=800,new_baseline_games=0,fixed_learning_rate=1e-5,qualified_checkpoint=str(checkpoint)))
        for band in bands:
            with (ROOT/f'runs/training_reservation_{band}.json').open('x') as file:json.dump(dict(run=str(out),start=band,stop_exclusive=band+1000000),file)
        write(out/'started.json',dict(time=time.time(),status='prepared_not_launched'))
        write(out/'goal_control.json',dict(status='prepared_review_pending',goal_status='active',automatic_learning=True,publication=False,
            next_action='One cumulative read-only review,then two fixedreferenceKL0.1 qualified500CPU PPO repeats.'))
        check_plan(original);check_plan(plan)
        print(json.dumps(dict(out=str(out),prepared=True,comparator_wins=500,new_baseline_games=0,parent_state_exact=True,byte_identical=True,
            minimum_steps=settings['minimum_steps'],maximum_steps=settings['maximum_steps'],frozen_files=len(plan['frozen_sha256']))))
    finally:
        for lock in locks:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();prepare(args.source.resolve(),args.out.resolve())
