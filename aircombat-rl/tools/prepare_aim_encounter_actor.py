"""Prepare an isolated encounter-policy intervention with authenticated controls."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import time

from experiments.rear_async.storage import identity,read,sha,verify,write
from experiments.aim_async.encounter_phase import RULE
from tools.prepare_aim_opening_coverage import completed_panel,assert_profile
from tools.rear_async_cpu import ROOT,check_plan

DESIGN='docs/superpowers/plans/2026-10-15-encounter-actor-isolation.md'
CONTRACT=dict(rule=RULE,frozen_opening=True,trainable_after_exit=True,history_feature_index=43,
    normalized_range_boundary='float32(1500/11500)',sampling_streams=1)
METADATA={'parent','source_run','seeds','training_bands','baseline_monitor_expected','scope',
    'intervention','budget_spec_repository_relative','frozen_sha256','config_id',
    'learner_feature_dim','frozen_actor_parameters'}


def assert_protocol(plan,previous):
    if set(plan)!=set(previous)|{'encounter_policy'}:
        raise ValueError('Unexpected plan fields')
    if plan['encounter_policy']!=CONTRACT or plan['learner_feature_dim']!=44 or plan['frozen_actor_parameters']!=7561:
        raise ValueError('Unexpected encounter architecture contract')
    for key in previous.keys()-METADATA:
        if plan[key]!=previous[key]:raise ValueError('Unintended factor change: '+key)


def encounter_plan(previous,out,seed,band):
    if type(seed) is not int or seed<0 or type(band) is not int or band<0:
        raise ValueError('Require nonnegative integer seed/band')
    plan=copy.deepcopy(previous)
    plan.update(parent=str((out/'unchanged_parent.zip').relative_to(ROOT)),
        source_run=str((ROOT/previous['parent']).parent),seeds=[seed],training_bands=[band],
        baseline_monitor_expected=str(out/'baseline_monitor_expected.json'),
        scope='Frozen qualified opening actor with independently trainable post-encounter actor',
        intervention=dict(factor='encounter_actor_isolation',before='one_trainable_actor',
            after='frozen_opening_and_trainable_suffix_same_initial_parent',
            unchanged='Original trainable NN/Adam/counters; native physics/controller/rewards/PPO/foes/prefix curriculum/evaluation',
            limitation='Episodes ending before a surviving first range exit cannot improve through this intervention'),
        budget_spec_repository_relative=DESIGN,encounter_policy=copy.deepcopy(CONTRACT),
        learner_feature_dim=44,frozen_actor_parameters=7561)
    assert_protocol(plan,previous)
    return plan


def transform_runner(code):
    replacements=[
        ('from tools.aim_opening_coverage_reporting import update','from tools.aim_encounter_reporting import update'),
        ('from tools.aim_cpu_export import export','from tools.aim_encounter_export import export'),
        ('from tools.league_residual_action_mode import build_variant','from tools.aim_encounter_export import build_variant'),
        ('from experiments.aim_async.opening_anchor import load_with_rate','from experiments.aim_async.encounter_actor import load_with_rate'),
        ('from experiments.aim_async.feature_train import enable_actor_features','from experiments.aim_async.encounter_actor import enable_actor_features'),
        ('from experiments.aim_async.reengagement import make_worker','from experiments.aim_async.encounter_worker import make_worker'),
        ('from experiments.aim_async.policy import AimPriorPolicy','from experiments.aim_async.encounter_actor import EncounterActorPolicy'),
        ('assert all(p.requires_grad for p in model.policy.parameters())',
         "from experiments.aim_async.encounter_actor import enable_actor_features\n        assert enable_actor_features(model.policy)['frozen_parameters']==7561"),
        ('==(0,6976,585,7041)','==(7561,6976,585,7041)'),
        ('type(model.policy) is AimPriorPolicy and model.observation_space.shape==(43,)',
         'type(model.policy) is EncounterActorPolicy and model.observation_space.shape==(44,)')]
    for old,new in replacements:
        if code.count(old)!=1:raise ValueError('Unexpected runner binding: '+old)
        code=code.replace(old,new)
    return code


def core_state(model):
    from tools.prepare_rear_cpu_entropy import tensor_digest
    return dict(policy=tensor_digest({k:v for k,v in model.policy.state_dict().items() if not k.startswith('opening_')}),
        optimizer=tensor_digest(model.policy.optimizer.state_dict()),decisions=model.num_timesteps,updates=model._n_updates)


def prepare(previous,evidence,out,seed,band):
    import psutil
    import torch
    from experiments.aim_async.opening_anchor import load_with_rate as original_load
    from experiments.aim_async.encounter_actor import load_with_rate
    from experiments.aim_async.evaluate import profile
    from tools.aim_known_qualification import index_panel
    from tools.aim_paired_regression_trace import no_live_native,assert_native_equal
    from tools.prepare_aim_known_long import terminal
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Require a fresh runs namespace')
    no_live_native()
    if psutil.virtual_memory().available<10*2**30:raise MemoryError('Need10GiB available')
    locks=[]
    try:
        for folder in (previous,evidence):
            for name in ('launch.lock','training.lock','evaluation.lock'):
                lock=(folder/name).open('a');locks.append(lock);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            terminal(folder);check_plan(read(folder/'plan.json'))
        old=read(previous/'plan.json');selected=Path(old['selected_parent_checkpoint']).resolve();meta=verify(selected)
        choice=read(evidence/'SELECTION_DECISION.json')
        if choice['promoted'] or Path(choice['selected_parent']).resolve()!=selected:
            raise ValueError('Qualified parent was not retained')
        diagnosis=read(evidence/'phase_gradient_diagnosis/completion.json')
        if diagnosis['optimizer_steps']!=0 or not diagnosis['model_tensors_unchanged']:
            raise ValueError('Diagnostic source mutated')
        if read(evidence/'CROSSED_PREFIX_REVIEW.json')['informative_discordant_cases']!=2:
            raise ValueError('Expected paired diagnostic evidence')
        q=previous/'qualification_s47232_step_010485760'
        lock=(q/'qualification.lock').open('a');locks.append(lock);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows54,evidence_files=completed_panel(q,selected,old)
        if profile(rows54)['wins']!=old['qualified_parent_all54_wins']:raise ValueError('All54 comparator changed')
        indexed={(r['opponent'],r['seed'],r['seat'],r['action_seed']):r for r in rows54}
        panels=[]
        for kind in ('monitor','development'):
            paths=[p for p in (previous/'evaluations').glob('*/completion.json') if read(p).get('baseline') and read(p)['kind']==kind]
            if len(paths)!=1:raise ValueError('Ambiguous baseline panel: '+kind)
            file=paths[0];entry=read(file);cells=sorted((file.parent/'cells').glob('*.json'));rows=[read(p) for p in cells]
            if Path(entry['checkpoint']).resolve()!=selected or entry['checkpoint_id']!=meta['checkpoint_id']:
                raise ValueError('Baseline checkpoint mismatch')
            samplers=old['monitor_action_seeds'] if kind=='monitor' else [4900,4901]
            index_panel(rows,[f['id'] for f in old[kind+'_opponents']],old[kind+'_band'],old[kind+'_ics'],samplers)
            assert_profile(profile(rows),entry['summary'])
            for row in rows:assert_native_equal(indexed[tuple(row[k] for k in ('opponent','seed','seat','action_seed'))],row)
            panels.append((file,entry,cells,rows))
        for key in ('monitor','development'):
            start=old[key+'_band'];stop=start+old[key+'_ics']
            if band<stop and band+1000000>start:raise ValueError('Training/evaluation overlap')
        for file in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            reservation=read(file)
            if band<reservation['stop_exclusive'] and band+1000000>reservation['start']:
                raise ValueError('Training band already reserved')
        torch.set_num_threads(1)
        source=original_load(selected/'learner.zip',env=None,seed=seed,learning_rate=old['ppo']['learning_rate'])
        restored=load_with_rate(selected/'learner.zip',env=None,seed=seed,learning_rate=old['ppo']['learning_rate'])
        before=core_state(source)
        if core_state(restored)!=before:raise ValueError('Core NN/Adam/counters changed')
        anchor=old['opening_anchor'];bank=ROOT/anchor['bank']
        if sha(bank)!=anchor['bank_sha256']:raise ValueError('Opening bank changed')
        restored.attach_opening_anchor(bank,anchor['coefficient'],anchor['batch_size'],seed)
        with torch.no_grad():
            obs=restored.opening_observations;expected=source.policy.get_distribution(obs[:,:43]).distribution.logits
            expected_values=source.policy.predict_values(obs[:,:43])
            for phase in (0,1):
                probe=obs.clone();probe[:,-1]=phase
                if not torch.equal(restored.policy.get_distribution(probe).distribution.logits,expected):raise ValueError('Initial logits differ')
                if not torch.equal(restored.policy.predict_values(probe),expected_values):raise ValueError('Initial value differs')
            initial_kl=float(restored.opening_kl_all())
        if abs(initial_kl)>1e-7:raise ValueError('Initial opening actor changed')
        out.mkdir(parents=True);prepared=out/'unchanged_parent.zip';shutil.copyfile(selected/'learner.zip',prepared)
        if sha(prepared)!=sha(selected/'learner.zip'):raise ValueError('Copied source changed')
        code=transform_runner((previous/'aim_cpu_runner.py').read_text());runner=out/'aim_cpu_runner.py'
        compile(code,str(runner),'exec');runner.write_text(code)
        plan=encounter_plan(old,out,seed,band);write(out/'baseline_monitor_expected.json',panels[0][3])
        files=[Path(__file__),runner,prepared,out/'baseline_monitor_expected.json',previous/'plan.json',previous/'aim_cpu_runner.py',
            ROOT.parent/DESIGN,*evidence_files]
        files.extend(ROOT/p for p in ('experiments/aim_async/encounter_phase.py','experiments/aim_async/encounter_actor.py',
            'experiments/aim_async/encounter_worker.py','tools/aim_encounter_export.py','tools/aim_encounter_reporting.py',
            'tests/test_encounter_actor.py','tests/test_encounter_runtime.py','tests/test_prepare_encounter_actor.py'))
        files.extend(evidence/p for p in ('SELECTION_DECISION.json','ALL54_COMPARISON_REVIEW.json','POST_CYCLE_DIAGNOSIS.md',
            'NEXT_INTERVENTION_REVIEW.md','REPEATED_REGRESSION_REVIEW.json','CROSSED_PREFIX_REVIEW.json','phase_gradient_diagnosis/completion.json'))
        for file,_,cells,_ in panels:files.extend([file,*cells])
        plan['frozen_sha256'].update({os.path.relpath(p.resolve(),ROOT):sha(p) for p in files})
        plan['config_id']=identity({k:v for k,v in plan.items() if k not in ('config_id','frozen_sha256')})
        assert_protocol(plan,old);write(out/'plan.json',plan)
        for file,entry,_,rows in panels:
            target=out/'evaluations'/identity(dict(config_id=plan['config_id'],kind=entry['kind']))
            cached=dict(entry,baseline=True,additional_steps=0,seed=seed,win_gain=None,wall_seconds=0.,actual_new_games=0,
                new_games_in_final_attempt=0,reused_cells_in_final_attempt=len(rows),reused_from_sibling_games=0,
                reused_from_run=str(previous),reused_from_evaluation=str(file.parent),
                reuse_basis='Authenticated qualified parent and unchanged native comparison conditions')
            write(target/'completion.json',cached);write(out/'candidates'/f'{target.name}.json',cached)
            for row in rows:write(target/'cells'/f"{identity({k:row[k] for k in ('opponent','seed','seat','action_seed')})}.json",row)
        write(out/'baseline_checkpoint.json',dict(path=str(selected)))
        write(out/'parent_verification.json',dict(before=before,after=core_state(restored),core_state_exact=True,
            byte_identical_source=True,initial_both_branch_logits_and_values_exact=True,bank_rows=len(obs),
            initial_bank_kl=initial_kl,prepared_sha256=sha(prepared),source_sha256=sha(selected/'learner.zip'),
            frozen_opening_parameters=7561,baseline_unique_games=672,baseline_panel_entries=800,new_baseline_games=0,parent_all54_wins=1310))
        reservation=ROOT/f'runs/training_reservation_{band}.json'
        with reservation.open('x') as stream:
            import json
            json.dump(dict(run=str(out),start=band,stop_exclusive=band+1000000),stream)
        write(out/'started.json',dict(time=time.time(),status='prepared_not_launched'))
        write(out/'goal_control.json',dict(status='prepared',goal_status='active',publication=False))
        check_plan(old);check_plan(plan)
        print(dict(prepared=True,seed=seed,core_state_exact=True,initial_bank_kl=initial_kl,
            baseline_new_games=0,additional_steps=plan['maximum_steps'],encounter_policy=plan['encounter_policy']))
    finally:
        for lock in locks:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('previous','evidence','out'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True);parser.add_argument('--band',type=int,required=True)
    args=parser.parse_args();prepare(args.previous.resolve(),args.evidence.resolve(),args.out.resolve(),args.seed,args.band)
