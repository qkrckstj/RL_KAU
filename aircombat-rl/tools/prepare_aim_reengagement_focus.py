"""Isolate native reengagement attempt frequency; preserve all frozen learning code."""
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
    if set(plan)!=set(previous):
        raise ValueError('Unexpected plan fields')
    if previous['curriculum'] != dict(stage='post_encounter',fraction=.5):
        raise ValueError('Expected completed half-prefix comparator')
    fraction=plan['curriculum'].get('fraction')
    if (type(fraction) not in (int,float) or fraction!=1. or
            plan['curriculum']!=dict(stage='post_encounter',fraction=1.)):
        raise ValueError('Expected all native-prefix attempts with original fallback')
    for key in previous.keys()-METADATA-{'curriculum'}:
        if plan[key]!=previous[key]:
            raise ValueError('Unintended factor change: '+key)


def focus_plan(previous, out, seed, band):
    if type(seed) is not int or seed<0 or type(band) is not int or band<0:
        raise ValueError('Require nonnegative integer seed and band')
    plan=copy.deepcopy(previous)
    plan.update(parent=str((out/'unchanged_parent.zip').relative_to(ROOT)),
        source_run=str((ROOT/previous['parent']).parent),seeds=[seed],training_bands=[band],
        baseline_monitor_expected=str(out/'baseline_monitor_expected.json'),
        scope='Native downstream PPO with all reengagement attempts and original fallback',
        intervention=dict(factor='native_reengagement_attempt_fraction',before=.5,after=1.,
            comparator='Completed s47232; fresh training RNG limits causal attribution',
            unchanged='Qualified NN/Adam/counters, PPO/rewards/foes/bank/prefix pool/boundary/evaluation/controller/physics'),
        budget_spec_repository_relative='docs/superpowers/plans/2026-10-15-reengagement-focus.md',
        curriculum=dict(stage='post_encounter',fraction=1.))
    assert_protocol(plan,previous)
    return plan


def transform_runner(code):
    old='from tools.aim_opening_coverage_reporting import update'
    if code.count(old)!=1:raise ValueError('Unexpected reporter binding')
    return code.replace(old,'from tools.aim_reengagement_focus_reporting import update')


def prepare(previous,out,seed,band):
    import psutil
    import torch
    from experiments.aim_async.evaluate import profile
    from experiments.aim_async.opening_anchor import load_with_rate
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
        terminal(previous)
        old=read(previous/'plan.json');check_plan(old)
        decision=read(previous/'NEXT_FACTOR_DECISION.json')
        if (decision['selected_factor']!='native_reengagement_attempt_fraction' or
                decision['before']!=.5 or decision['after']!=1. or
                decision['seed']!=seed or decision['training_band']!=band or
                ROOT/decision['new_run']!=out):
            raise ValueError('Preparation differs from evidence-backed factor decision')
        selected=Path(old['selected_parent_checkpoint']).resolve();meta=verify(selected)
        selection=read(previous/'SELECTION_DECISION.json')
        if selection['promotion'] or Path(selection['selected_checkpoint']).resolve()!=selected:
            raise ValueError('Qualified parent was not retained')
        audit=read(previous/'FINAL_TRAINING_REVIEW.json')
        if not (audit['all_512_ppo_rows_finite'] and audit['all_episode_metrics_finite'] and
                audit['final_policy_and_optimizer_finite'] and audit['terminal_health_verdict_mismatches']==0):
            raise ValueError('Completed trial audit is not clean')
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
        source=load_with_rate(selected/'learner.zip',env=None,seed=seed,learning_rate=old['ppo']['learning_rate'])
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
        plan=focus_plan(old,out,seed,band)
        write(out/'baseline_monitor_expected.json',panels[0][3])
        files=[Path(__file__),ROOT/'tools/aim_reengagement_focus_reporting.py',ROOT/'tests/test_reengagement_focus.py',
            ROOT.parent/'docs/superpowers/plans/2026-10-15-reengagement-focus.md',
            previous/'plan.json',previous/'aim_cpu_runner.py',prepared,runner,out/'baseline_monitor_expected.json',
            previous/'NEXT_FACTOR_DECISION.json',previous/'FINAL_TRAINING_REVIEW.json',previous/'SELECTION_DECISION.json',
            previous/'ALL54_COMPARISON_REVIEW.json',previous/'CROSSED_PREFIX_REVIEW.json',
            previous/'handoff_diagnosis_v2/completion.json',*evidence_files]
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
                   baseline_new_games=0,additional_steps=plan['maximum_steps'],attempt_fraction=1.))
    finally:
        for lock in locks:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('previous','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--band',type=int,required=True)
    a=p.parse_args();prepare(a.previous.resolve(),a.out.resolve(),a.seed,a.band)
