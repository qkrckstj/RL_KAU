"""Continue an authenticated encounter policy with one opponent weight change."""
import argparse
import copy
import fcntl
import math
import os
from pathlib import Path
import shutil
import time

from experiments.rear_async.storage import identity,read,sha,verify,write
from tools.rear_async_cpu import ROOT,check_plan
from tools.prepare_aim_opening_coverage import completed_panel,assert_profile

TARGET='untrained_parameter_01'
DESIGN='docs/superpowers/plans/2026-10-15-encounter-loss-focus.md'
METADATA={'parent','parent_source','source_run','seeds','training_bands','source_actor_feature_sha256',
    'qualified_parent_full_wins','qualified_parent_full_games','qualified_parent_all54_wins','qualified_parent_all54_games',
    'selected_parent_checkpoint','selected_parent_assessment','baseline_monitor_expected','scope','intervention',
    'comparison_scope','baseline_all54_qualification','baseline_all54_checkpoint','budget_spec_repository_relative',
    'parent_choice','frozen_sha256','config_id'}

def focused_probabilities(opponents,probabilities):
    ids=[o['id'] for o in opponents]
    if len(ids)!=54 or len(set(ids))!=54 or TARGET not in ids or len(probabilities)!=54:
        raise ValueError('Require the complete unique54 roster')
    if any(not math.isfinite(x) or x<=0 for x in probabilities) or not math.isclose(sum(probabilities),1.,abs_tol=1e-12,rel_tol=0):
        raise ValueError('Require positive normalized sampling probabilities')
    result=list(probabilities);result[ids.index(TARGET)]*=2
    total=sum(result)
    return [x/total for x in result]

def reserve_training_band(out,band,registry):
    """Serialize overlap validation/publication for cooperating local preparers."""
    if type(band) is not int or band<0:raise ValueError('Nonnegative integer band required')
    with (registry/'.training_reservations.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for file in registry.glob('**/training_reservation_*.json'):
            item=read(file)
            if band<item['stop_exclusive'] and band+1000000>item['start']:
                raise ValueError('Training band already reserved')
        reservation=out.parent/f'training_reservation_{band}.json'
        with reservation.open('x') as stream:
            import json
            json.dump(dict(run=str(out),start=band,stop_exclusive=band+1000000),stream)
            stream.flush();os.fsync(stream.fileno())

def assert_protocol(plan,previous):
    if set(plan)!=set(previous)|{'qualified_parent_full_games'}:
        raise ValueError('Unexpected plan schema')
    for key in previous.keys()-METADATA-{'probabilities'}:
        if plan[key]!=previous[key]:raise ValueError('Unintended factor change: '+key)
    if plan['probabilities']!=focused_probabilities(previous['opponents'],previous['probabilities']):
        raise ValueError('Unexpected allocation change')

def continuation_plan(previous,source,out,checkpoint,metadata,seed,band):
    from tools.prepare_aim_opening_coverage import continuation_plan as base
    if type(seed) is not int or seed<0 or type(band) is not int or band<0:
        raise ValueError('Nonnegative integer seed/band required')
    qualification=source/'qualification_s47236_step_002621440'
    plan=base(previous,(out/'unchanged_parent.zip').relative_to(ROOT),source,seed,band,metadata,
        508,1319,checkpoint,qualification/'completion.json',out/'baseline_monitor_expected.json')
    plan.update(parent_source=str(checkpoint/'learner.zip'),
        probabilities=focused_probabilities(previous['opponents'],previous['probabilities']),
        baseline_all54_qualification=str(qualification),baseline_all54_checkpoint=str(checkpoint),
        parent_choice='Highest verified originalCPU all54 wins; unchanged original weakest14 total; preserve previous reference and all tradeoff candidates.',
        budget_spec_repository_relative=DESIGN,
        scope='Selected encounter policy continuation with one loss-heavy opponent weight doubled',
        intervention=dict(factor='double_one_loss_heavy_opponent_sampling_weight',opponent=TARGET,multiplier=2.,
            formula='Multiply target probability by2 then normalize all54; periodic coverage unchanged',
            evidence=str(source/'POST_CYCLE_ALLOCATION_REVIEW.json'),
            unchanged='Selected full NN/Adam/counters/frozen opening; PPO/rewards/physics/controller/prefix pool/coverage/evaluation'))
    assert_protocol(plan,previous)
    return plan

def restoration_evidence(checkpoint,plan,seed):
    import torch
    from experiments.aim_async.encounter_actor import EncounterPPO,load_with_rate,enable_actor_features
    from tools.prepare_rear_cpu_entropy import tensor_digest
    torch.set_num_threads(1);verify(checkpoint)
    source=EncounterPPO.load(checkpoint/'learner.zip',env=None,device='cpu')
    restored=load_with_rate(checkpoint/'learner.zip',env=None,seed=seed,learning_rate=plan['ppo']['learning_rate'])
    def state(m):return dict(policy=tensor_digest(m.policy.state_dict()),optimizer=tensor_digest(m.policy.optimizer.state_dict()),decisions=m.num_timesteps,updates=m._n_updates)
    before=state(source);after=state(restored)
    if before!=after:raise ValueError('Full NN/Adam/counters changed on restore')
    anchor=plan['opening_anchor'];bank=ROOT/anchor['bank']
    if sha(bank)!=anchor['bank_sha256']:raise ValueError('Opening bank changed')
    restored.attach_opening_anchor(bank,anchor['coefficient'],anchor['batch_size'],seed)
    distinct=False
    with torch.no_grad():
        obs=restored.opening_observations
        opening=restored.policy.get_distribution(obs).distribution.logits.clone()
        for phase in (0,1):
            probe=obs.clone();probe[:,-1]=phase
            expected=source.policy.get_distribution(probe).distribution.logits
            actual=restored.policy.get_distribution(probe).distribution.logits
            if not torch.equal(actual,expected) or not torch.equal(restored.policy.predict_values(probe),source.policy.predict_values(probe)):
                raise ValueError('Branch logits/value changed on restore')
            if phase==1:distinct=not torch.equal(actual,opening)
        kl=float(restored.opening_kl_all())
    if not distinct or abs(kl)>1e-7:raise ValueError('Expected learned suffix distinct from preserved opening')
    roles=enable_actor_features(restored.policy)
    if roles['frozen_parameters']!=7561:raise ValueError('Frozen opening roles changed')
    return dict(before=before,after=after,core_state_exact=True,full_state_exact=True,
        initial_both_branch_logits_and_values_exact=True,branches_distinct=True,
        comparison='Each restored branch equals its corresponding saved branch; branches need not equal each other.',
        initial_bank_kl=kl,bank_rows=len(obs),frozen_opening_parameters=7561,
        source_sha256=sha(checkpoint/'learner.zip'))

def prepare(source,out,seed,band):
    import psutil
    from experiments.aim_async.evaluate import profile
    from experiments.league.health_rule_v2 import classify
    from tools.aim_known_qualification import index_panel
    from tools.aim_paired_regression_trace import no_live_native,assert_native_equal
    from tools.prepare_aim_known_long import terminal
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh local run required')
    no_live_native()
    if psutil.virtual_memory().available<10*2**30:raise MemoryError('Need10GiB available')
    selection=read(source/'SELECTION_DECISION.json');selected=Path(selection['selected_checkpoint']).resolve()
    if not selection['promoted'] or not selected.is_relative_to(source/'checkpoints'):
        raise ValueError('Expected locally selected source checkpoint')
    q=source/'qualification_s47236_step_002621440';locks=[]
    try:
        for folder,names in ((source,('launch.lock','training.lock','evaluation.lock')),(q,('qualification.lock',))):
            for name in names:
                lock=(folder/name).open('a');locks.append(lock);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        terminal(source);old=read(source/'plan.json');check_plan(old);meta=verify(selected)
        rows54,files=completed_panel(q,selected,old)
        if profile(rows54)['wins']!=1319 or selection['selected_outcomes']['win']!=1319:
            raise ValueError('Selected all54 evidence differs')
        if any(classify(row['own_health'],row['opp_health'])!=row['result'] for row in rows54):
            raise ValueError('Cached health verdict mismatch')
        indexed={tuple(row[k] for k in ('opponent','seed','seat','action_seed')):row for row in rows54}
        panels=[]
        for kind in ('monitor','development'):
            matches=[f for f in (source/'evaluations').glob('*/completion.json') if read(f)['kind']==kind and Path(read(f)['checkpoint']).resolve()==selected]
            if len(matches)!=1:raise ValueError('Ambiguous selected panel: '+kind)
            file=matches[0];entry=read(file);cells=sorted((file.parent/'cells').glob('*.json'));rows=[read(f) for f in cells]
            if entry['baseline'] or entry['checkpoint_id']!=meta['checkpoint_id']:raise ValueError('Wrong selected panel identity')
            samplers=old['monitor_action_seeds'] if kind=='monitor' else [4900,4901]
            index_panel(rows,[o['id'] for o in old[kind+'_opponents']],old[kind+'_band'],old[kind+'_ics'],samplers)
            assert_profile(profile(rows),entry['summary'])
            if entry['summary']['wins']!=({'monitor':60,'development':508}[kind]):raise ValueError('Selected subset score changed')
            for row in rows:assert_native_equal(indexed[tuple(row[k] for k in ('opponent','seed','seat','action_seed'))],row)
            panels.append((file,entry,rows));files.extend([file,*cells])
        for kind in ('monitor','development'):
            if band<old[kind+'_band']+old[kind+'_ics'] and band+1000000>old[kind+'_band']:
                raise ValueError('Training/evaluation overlap')
        evidence=restoration_evidence(selected,old,seed)
        out.mkdir(parents=True);prepared=out/'unchanged_parent.zip';shutil.copyfile(selected/'learner.zip',prepared)
        if sha(prepared)!=evidence['source_sha256']:raise ValueError('Parent copy differs')
        runner=out/'aim_cpu_runner.py';code=(source/'aim_cpu_runner.py').read_text()
        old_binding='from tools.aim_encounter_reporting_v2 import update'
        if code.count(old_binding)!=1:raise ValueError('Unexpected runner reporter binding')
        code=code.replace(old_binding,'from tools.aim_encounter_focus_reporting import update')
        compile(code,str(runner),'exec');runner.write_text(code)
        plan=continuation_plan(old,source,out,selected,meta,seed,band)
        write(out/'baseline_monitor_expected.json',panels[0][2])
        files.extend([Path(__file__),runner,prepared,out/'baseline_monitor_expected.json',source/'plan.json',source/'aim_cpu_runner.py',ROOT.parent/DESIGN,
            ROOT/'tools/aim_encounter_focus_reporting.py',ROOT/'tests/test_prepare_encounter_focus.py'])
        files.extend(source/name for name in ('SELECTION_DECISION.json','EARLY_ALL54_COMPARISON_REVIEW.json','NEXT_INTERVENTION_REVIEW.json','POST_CYCLE_ALLOCATION_REVIEW.json','actor_signal_diagnosis/completion.json'))
        files.extend([selected/'READY.json',*(selected/name for name in meta['files'])])
        plan['frozen_sha256'].update({os.path.relpath(f.resolve(),ROOT):sha(f) for f in files})
        plan['config_id']=identity({k:v for k,v in plan.items() if k not in ('config_id','frozen_sha256')})
        assert_protocol(plan,old);write(out/'plan.json',plan)
        for file,entry,rows in panels:
            target=out/'evaluations'/identity(dict(config_id=plan['config_id'],kind=entry['kind']))
            cached=dict(entry,baseline=True,additional_steps=0,seed=seed,win_gain=None,wall_seconds=0.,actual_new_games=0,
                new_games_in_final_attempt=0,reused_cells_in_final_attempt=len(rows),reused_from_sibling_games=0,
                improvement_confirmed=False,reused_from_run=str(source),reused_from_evaluation=str(file.parent),
                reuse_basis='Exact selected encounter checkpoint and unchanged native comparison conditions')
            write(target/'completion.json',cached);write(out/'candidates'/f'{target.name}.json',cached)
            for row in rows:write(target/'cells'/f"{identity({k:row[k] for k in ('opponent','seed','seat','action_seed')})}.json",row)
        write(out/'baseline_checkpoint.json',dict(path=str(selected)))
        write(out/'parent_verification.json',dict(evidence,byte_identical_source=True,prepared_sha256=sha(prepared),baseline_unique_games=672,baseline_panel_entries=800,new_baseline_games=0,parent_all54_wins=1319))
        reserve_training_band(out,band,ROOT.parent.parent.parent)
        write(out/'started.json',dict(time=time.time(),status='prepared_not_launched'))
        write(out/'goal_control.json',dict(status='prepared',goal_status='active',publication=False))
        check_plan(plan)
        from tools.aim_encounter_focus_reporting import update
        report=update(out)
        if report['new_evaluation_games']!=0 or report['baseline_unique_games']!=672:raise ValueError('Baseline accounting differs')
        print(dict(prepared=True,seed=seed,parent_all54_wins=1319,full_state_exact=True,new_baseline_games=0,
            target_probability=plan['probabilities'][[o['id'] for o in plan['opponents']].index(TARGET)]))
    finally:
        for lock in locks:lock.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('source','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True);parser.add_argument('--band',type=int,required=True)
    args=parser.parse_args();prepare(args.source.resolve(),args.out.resolve(),args.seed,args.band)
