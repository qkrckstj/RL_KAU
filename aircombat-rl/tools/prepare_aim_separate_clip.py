"""Prepare an isolated actor/critic clipping trial from the completed half-prefix control."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import time

from experiments.rear_async.storage import identity, read, sha, verify, write
from tools.prepare_aim_opening_coverage import assert_profile, completed_panel
from tools.rear_async_cpu import ROOT, check_plan


METADATA = {'parent','source_run','seeds','training_bands','baseline_monitor_expected',
            'scope','intervention','budget_spec_repository_relative','frozen_sha256','config_id'}


def assert_protocol(plan, previous):
    if set(plan) != set(previous) | {'gradient_clipping'}:
        raise ValueError('Unexpected plan fields')
    if previous['curriculum'] != dict(stage='post_encounter',fraction=.5):
        raise ValueError('Expected completed half-prefix control')
    if plan['gradient_clipping'] != dict(mode='separate_actor_critic',actor_max_norm=.5,critic_max_norm=.5):
        raise ValueError('Unexpected clipping configuration')
    if previous['ppo']['max_grad_norm'] != .5:
        raise ValueError('Expected original norm0.5')
    for key in previous.keys()-METADATA:
        if plan[key] != previous[key]:
            raise ValueError('Unintended factor change: '+key)


def clipping_plan(previous, out, seed, band):
    if type(seed) is not int or seed<0 or type(band) is not int or band<0:
        raise ValueError('Require nonnegative integer seed and band')
    plan=copy.deepcopy(previous)
    plan.update(parent=str((out/'unchanged_parent.zip').relative_to(ROOT)),
        source_run=str((ROOT/previous['parent']).parent),seeds=[seed],training_bands=[band],
        baseline_monitor_expected=str(out/'baseline_monitor_expected.json'),
        scope='Separate actor/critic clipping; original native half-prefix control',
        intervention=dict(factor='actor_critic_gradient_clipping',before='joint0.5',after='separate0.5',
            comparator='Completed s47232; failed s47233 fraction1 reverted; fresh RNG limits attribution',
            unchanged='Qualified NN/Adam/counters, PPO losses/rewards/foes/bank/prefix pool/boundary/evaluation/controller/physics'),
        budget_spec_repository_relative='docs/superpowers/plans/2026-10-15-separate-gradient-clipping.md',
        gradient_clipping=dict(mode='separate_actor_critic',actor_max_norm=.5,critic_max_norm=.5))
    assert_protocol(plan,previous)
    return plan


def transform_runner(code):
    for old,new in [
        ('from tools.aim_opening_coverage_reporting import update','from tools.aim_separate_clip_reporting import update'),
        ('from experiments.aim_async.opening_anchor import load_with_rate','from experiments.aim_async.separate_clipping import load_with_rate')]:
        if code.count(old)!=1:raise ValueError('Unexpected runner binding: '+old)
        code=code.replace(old,new)
    line="                model.attach_reference(ROOT/plan['reference_archive'],plan['reference_kl_coefficient'])"
    if code.count(line)!=1:raise ValueError('Unexpected reference attachment boundary')
    code=code.replace(line,"                assert type(model).__name__=='SeparateClippedOpeningPPO'\n"+line)
    return code


def prepare(previous,evidence,out,seed,band):
    import psutil
    import torch
    from experiments.aim_async.evaluate import profile
    from experiments.aim_async.opening_anchor import load_with_rate as original_load
    from experiments.aim_async.separate_clipping import load_with_rate
    from tools.aim_prior_bias import learner_state
    from tools.aim_known_qualification import index_panel
    from tools.aim_paired_regression_trace import no_live_native
    from tools.prepare_aim_known_long import terminal
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Require a fresh runs namespace')
    no_live_native()
    if psutil.virtual_memory().available<10*2**30:raise MemoryError('Need10GiB available')
    locks=[]
    try:
        for name in ('launch.lock','training.lock','evaluation.lock'):
            lock=(previous/name).open('a');locks.append(lock);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        terminal(previous);terminal(evidence)
        check_plan(read(evidence/'plan.json'))
        old=read(previous/'plan.json');check_plan(old)
        decision=read(evidence/'NEXT_FACTOR_DECISION.json')
        if (decision['selected_factor']!='actor_critic_gradient_clipping' or
                decision['before']!='joint_global_norm0.5' or decision['after']!='separate_actor_norm0.5_and_critic_norm0.5' or
                decision['seed']!=seed or decision['training_band']!=band or
                ROOT/decision['new_run']!=out):
            raise ValueError('Preparation differs from evidence-backed factor decision')
        selected=Path(old['selected_parent_checkpoint']).resolve();meta=verify(selected)
        selection=read(evidence/'SELECTION_DECISION.json')
        if selection['promoted'] or Path(selection['selected_parent']).resolve()!=selected:
            raise ValueError('Qualified parent was not retained')
        audit=read(evidence/'ENDPOINT_TRAINING_REVIEW.json')
        if not audit['finite_policy_optimizer_and_metrics'] or audit['additional_steps']!=10485760:
            raise ValueError('Completed diagnostic source is not clean')
        if read(evidence/'gradient_diagnosis/completion.json')['optimizer_steps']!=0:
            raise ValueError('Diagnostic mutated its model')
        q=previous/'qualification_s47232_step_010485760'
        lock=(q/'qualification.lock').open('a');locks.append(lock);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows54,evidence_files=completed_panel(q,selected,old)
        if profile(rows54)['wins']!=old['qualified_parent_all54_wins']:
            raise ValueError('All54 comparator changed')
        index54={(r['opponent'],r['seed'],r['seat'],r['action_seed']):r for r in rows54}
        panels=[]
        for kind in ('monitor','development'):
            paths=[p for p in (previous/'evaluations').glob('*/completion.json') if
                   read(p).get('baseline') and read(p)['kind']==kind]
            if len(paths)!=1:raise ValueError('Ambiguous baseline: '+kind)
            file=paths[0];entry=read(file);cells=sorted((file.parent/'cells').glob('*.json'));rows=[read(p) for p in cells]
            if Path(entry['checkpoint']).resolve()!=selected or entry['checkpoint_id']!=meta['checkpoint_id']:
                raise ValueError('Baseline checkpoint mismatch')
            samplers=old['monitor_action_seeds'] if kind=='monitor' else [4900,4901]
            index_panel(rows,[f['id'] for f in old[kind+'_opponents']],old[kind+'_band'],old[kind+'_ics'],samplers)
            assert_profile(profile(rows),entry['summary'])
            for row in rows:
                expected=index54[(row['opponent'],row['seed'],row['seat'],row['action_seed'])]
                if {k:v for k,v in row.items() if k!='wall_seconds'}!={k:v for k,v in expected.items() if k!='wall_seconds'}:
                    raise ValueError('Baseline differs from authenticated all54 cell')
            panels.append((file,entry,cells,rows))
        for key in ('monitor','development'):
            start=old[key+'_band'];stop=start+old[key+'_ics']
            if band<stop and band+1000000>start:raise ValueError('Training/evaluation bands overlap')
        for file in ROOT.parent.parent.parent.glob('**/training_reservation_*.json'):
            reservation=read(file)
            if band<reservation['stop_exclusive'] and band+1000000>reservation['start']:
                raise ValueError('Training band already reserved')
        torch.set_num_threads(1)
        source=original_load(selected/'learner.zip',env=None,seed=seed,learning_rate=old['ppo']['learning_rate'])
        before=learner_state(source)
        anchor=old['opening_anchor'];bank=ROOT/anchor['bank']
        if sha(bank)!=anchor['bank_sha256']:raise ValueError('Opening bank changed')
        source.attach_opening_anchor(bank,anchor['coefficient'],anchor['batch_size'],seed)
        with torch.no_grad(): initial_kl=float(source.opening_kl_all())
        if abs(initial_kl)>1e-7:raise ValueError('Opening bank initial policy changed')
        out.mkdir(parents=True)
        prepared=out/'unchanged_parent.zip';shutil.copyfile(selected/'learner.zip',prepared)
        restored=load_with_rate(prepared,env=None,seed=seed,learning_rate=old['ppo']['learning_rate'])
        if sha(prepared)!=sha(selected/'learner.zip') or learner_state(restored)!=before:
            raise ValueError('Initial policy/Adam/counters changed')
        code=transform_runner((previous/'aim_cpu_runner.py').read_text())
        runner=out/'aim_cpu_runner.py';compile(code,str(runner),'exec');runner.write_text(code)
        plan=clipping_plan(old,out,seed,band)
        write(out/'baseline_monitor_expected.json',panels[0][3])
        files=[Path(__file__),ROOT/'tools/aim_separate_clip_reporting.py',ROOT/'tests/test_prepare_separate_clipping.py',
            ROOT/'tests/test_separate_clipping.py',ROOT/'experiments/aim_async/separate_clipping.py',
            ROOT.parent/'docs/superpowers/plans/2026-10-15-separate-gradient-clipping.md',
            previous/'plan.json',previous/'aim_cpu_runner.py',prepared,runner,out/'baseline_monitor_expected.json',
            previous/'NEXT_FACTOR_DECISION.json',previous/'FINAL_TRAINING_REVIEW.json',previous/'SELECTION_DECISION.json',
            previous/'ALL54_COMPARISON_REVIEW.json',previous/'CROSSED_PREFIX_REVIEW.json',
            previous/'handoff_diagnosis_v2/completion.json',*evidence_files]
        files.extend(evidence/name for name in ('NEXT_FACTOR_DECISION.json','FINAL_COMPARISON_REVIEW.json','ENDPOINT_TRAINING_REVIEW.json','SELECTION_DECISION.json','SAVED_POLICY_SHIFT_REVIEW.json','GRADIENT_CLIPPING_COUNTERFACTUAL.json','gradient_diagnosis/completion.json'))
        for file,_,cells,_ in panels:files.extend([file,*cells])
        plan['frozen_sha256'].update({os.path.relpath(p.resolve(),ROOT):sha(p) for p in files})
        plan['config_id']=identity({k:v for k,v in plan.items() if k not in ('config_id','frozen_sha256')})
        assert_protocol(plan,old);write(out/'plan.json',plan)
        for file,entry,_,rows in panels:
            target=out/'evaluations'/identity(dict(config_id=plan['config_id'],kind=entry['kind']))
            cached=dict(entry,baseline=True,additional_steps=0,seed=seed,win_gain=None,wall_seconds=0.,actual_new_games=0,
                new_games_in_final_attempt=0,reused_cells_in_final_attempt=len(rows),reused_from_sibling_games=0,
                reused_from_run=str(previous),reused_from_evaluation=str(file.parent),
                reuse_basis='Exact qualified checkpoint and unchanged normal-start CPU cells')
            write(target/'completion.json',cached);write(out/'candidates'/f'{target.name}.json',cached)
            for row in rows:
                key={k:row[k] for k in ('opponent','seed','seat','action_seed')}
                write(target/'cells'/f'{identity(key)}.json',row)
        write(out/'baseline_checkpoint.json',dict(path=str(selected)))
        write(out/'parent_verification.json',dict(before=before,after=learner_state(restored),state_exact=True,
            byte_identical=True,initial_native_behavior_preserved=True,initial_bank_kl=initial_kl,
            prepared_sha256=sha(prepared),source_sha256=sha(selected/'learner.zip'),baseline_unique_games=672,
            baseline_panel_entries=800,new_baseline_games=0,parent_all54_wins=1310))
        reservation=ROOT/f'runs/training_reservation_{band}.json'
        with reservation.open('x') as f:
            import json
            json.dump(dict(run=str(out),start=band,stop_exclusive=band+1000000),f)
        write(out/'started.json',dict(time=time.time(),status='prepared_not_launched'))
        write(out/'goal_control.json',dict(status='prepared',goal_status='active',publication=False))
        check_plan(old);check_plan(plan)
        print(dict(prepared=True,seed=seed,initial_state_exact=True,initial_bank_kl=initial_kl,
                   baseline_new_games=0,additional_steps=plan['maximum_steps'],attempt_fraction=.5,clipping=plan['gradient_clipping']))
    finally:
        for lock in locks:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('previous','evidence','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--band',type=int,required=True)
    a=p.parse_args();prepare(a.previous.resolve(),a.evidence.resolve(),a.out.resolve(),a.seed,a.band)
