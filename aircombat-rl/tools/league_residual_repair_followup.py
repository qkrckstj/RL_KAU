"""Conditionally assess omitted repair-capable PPO checkpoints after training.

Only stdlib while waiting. Candidate choice uses development records only;
completed selection/final outcome values never rank alternative checkpoints.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import ctypes
from ctypes import wintypes
import os
import shutil
import sys
import time
import traceback

from tools import league_thread_benchmark as io


def eligible_status(status):
    if status=='final_profile_passed':return False
    if status in ('no_development_improvement','selection_failed','final_profile_failed'):return True
    raise ValueError('Unexpected predecessor status; inspect before any evaluation')


def select_omitted(records,excluded_ids):
    """At most one eligible snapshot per RNG; deduplicate exact weight artifacts."""
    chosen=[];seen=set()
    for seed in sorted({r['seed'] for r in records}):
        candidates=[r for r in records if r['seed']==seed and r['repair_passed'] and r['spec']['id'] not in excluded_ids]
        candidates.sort(key=lambda r:(tuple(r['rank']),-r['step']),reverse=True)
        for row in candidates:
            if row['weights_sha256'] not in seen:
                chosen.append(row);seen.add(row['weights_sha256']);break
    return chosen


def prepare(source,out,pid):
    if (out/'queue_plan.json').exists() or (out/'plan.json').exists():
        raise FileExistsError('Use a fresh follow-up folder')
    old=io.read(source/'plan.json');io.verify(old)
    runtime=io.read(source/'runtime.json');identity=io.read(source/'observed_process_identity.json')
    if runtime['pid']!=pid or identity['pid']!=pid:raise ValueError('Wrong predecessor identity')
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Set pre-import thread limits')
    sources=dict(old['source_sha256']);inputs=dict(old['input_sha256'])
    for name in ('tools/league_residual_repair_followup.py','tests/test_league_residual_repair_followup.py'):
        sources[name]=io.sha(io.ROOT/name)
    for path in (source/'plan.json',source/'observed_process_identity.json'):
        inputs[io.relative(path)]=io.sha(path)
    plan=dict(source=io.relative(source),after_pid=pid,created_filetime=identity['created_filetime'],
        workers=4,selection_band=69010000,selection_n=40,final_band=70000000,final_n=80,
        heldout_band=71000000,heldout_n=80,environment=io.environment(),source_sha256=sources,input_sha256=inputs,
        candidate_rule='After both independent continuations finish: among all completed checkpoints, apply the unchanged repair gate on the24-foe development panel. Exclude each main-search selected policy. Choose highest existing development rank among remaining eligible snapshots per RNG, at most1 per RNG/2 total; earlier step wins exact rank ties. Deduplicate exact weight artifacts. No selection/final outcome values used to rank snapshots.',
        opponent_rule='Preserve the72-foe archive and add main-search selected PPO policies plus chosen omitted candidates; use the same frozen roster and shared ICs/both seats for all comparison models.',
        eligibility='Skip when main final_profile_passed; otherwise only after clean completed no_development_improvement/selection_failed/final_profile_failed. Skip if no eligible omitted snapshot. Failed/incomplete predecessor requires inspection.',
        assessment_rule='Reuse the existing aggregate and temporal repair criteria, same final IC-cluster confidence rule, then unused temporal-parameter holdout only if final passes. New selection/final IC bands. No additional policy learning in this stage.',
        scope='Separately declared conditional follow-up for development snapshots omitted by aggregate ranking. Preserve original experiment and failures. No broad unseen-opponent/tournament claim or GitHub publication.')
    io.verify(plan);io.write(out/'queue_plan.json',plan)
    for band in (plan['selection_band'],plan['final_band'],plan['heldout_band']):
        path=io.ROOT/f'runs/residual_repair_reservation_{band}.json'
        reservation=dict(out=io.relative(out),band=band,queue_plan_sha256=io.sha(out/'queue_plan.json'))
        if path.exists():raise FileExistsError('Band already reserved: '+str(band))
        io.write(path,reservation)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(io.ROOT/name,target)
    return plan


def wait_for_source(plan,out):
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD
    k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    k.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=k.OpenProcess(0x100000|0x1000,False,plan['after_pid'])
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        values=[wintypes.FILETIME() for _ in range(4)]
        if not k.GetProcessTimes(handle,*(ctypes.byref(v) for v in values)):raise ctypes.WinError(ctypes.get_last_error())
        created=(values[0].dwHighDateTime<<32)+values[0].dwLowDateTime
        if created!=plan['created_filetime']:raise ValueError('PID was reused or predecessor changed')
        io.write(out/'queue_runtime.json',dict(pid=os.getpid(),after_pid=plan['after_pid'],
            created_filetime=created,started_at=datetime.now(timezone.utc).isoformat(),
            simulation_workers_while_waiting=0,numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules,own_memory=io.own_memory()))
        print('Waiting on verified native learner handle; zero simulation workers.',flush=True)
        while True:
            status=k.WaitForSingleObject(handle,45000)
            if status==0:break
            if status!=258:raise ctypes.WinError(ctypes.get_last_error())
        code=wintypes.DWORD()
        if not k.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
        io.write(out/'predecessor_exit.json',dict(pid=plan['after_pid'],exit_code=code.value,
            observed_at=datetime.now(timezone.utc).isoformat()))
        if code.value!=0:raise RuntimeError('Predecessor exited unsuccessfully; no automatic retry')
    finally:k.CloseHandle(handle)
    io.verify(plan)
    if plan['environment']!=io.environment():raise ValueError('Environment changed while waiting')


def build_assessment(source,out,queued,completion):
    from tools import league_residual_train as base
    from tools.league_temporal_repair_train import decision
    old=io.read(source/'plan.json');io.verify(old)
    if completion.get('heldout_opened'):raise ValueError('Do not reuse a consumed parameter holdout as unseen')
    if {s['seed'] for s in completion['searches']}!=set(old['seeds']):
        raise ValueError('Both independent continuation results are required')
    baseline_path=io.ROOT/old['previous']/'development_baseline.json'
    baseline=io.read(baseline_path)
    dev_groups={s['id']:old['groups'][s['id']] for s in old['development_opponents']}
    subset=dict(old,groups=dev_groups)
    base.validate_records(baseline['results'],old['development_opponents'],old['development_band'],old['development_n'])
    excluded={s['selected']['spec']['id'] for s in completion['searches']}
    main_selected=[s['selected']['spec'] for s in completion['searches']]
    records=[];paths=[source/'completion.json',baseline_path]
    for search in completion['searches']:
        result_path=source/f"s{search['seed']}/result.json"
        if io.read(result_path)!=search:raise ValueError('Completed search/result mismatch')
        paths.append(result_path)
        for chunk in search['history']:
            path=io.ROOT/chunk['development'];record=io.read(path)
            request=dict(spec=chunk['candidate']['spec'],opponents=old['development_opponents'],
                         band=old['development_band'],n=old['development_n'])
            if record['request']!=request:raise ValueError('Changed development schedule')
            base.validate_records(record['results'],old['development_opponents'],old['development_band'],old['development_n'])
            rank=list(base.ranking(base.profile(record['results'],dev_groups)))
            if record['rank']!=rank or chunk['candidate']['rank']!=rank:raise ValueError('Development rank mismatch')
            spec=chunk['candidate']['spec'];comparison=decision(record,baseline,subset)
            ck_path=io.ROOT/chunk['checkpoint']/'checkpoint.json';ck=io.read(ck_path)
            if ck['entrant']!=spec or ck['step']!=chunk['steps']:raise ValueError('Wrong checkpoint')
            paths += [path,ck_path]
            records.append(dict(seed=search['seed'],step=chunk['steps'],spec=spec,rank=rank,
                repair_passed=comparison['repair_passed'],development_comparison=comparison,
                weights_sha256=io.sha(io.ROOT/spec['weights'])))
    chosen=select_omitted(records,excluded)
    io.write(out/'development_candidate_review.json',dict(records=records,excluded_main_selected_ids=sorted(excluded),
        chosen=chosen,candidate_rule=queued['candidate_rule'],selection_or_final_outcome_values_used=False))
    if not chosen:return None
    opponents=base.unique_entrants(old['opponents']+main_selected+[r['spec'] for r in chosen])
    groups=base.code_groups(opponents)
    plan=deepcopy(old)
    plan.update(previous=io.relative(source),workers=4,opponents=opponents,groups=groups,
        probabilities=[1/len(opponents)]*len(opponents),
        opponent_mixture='Evaluation only; no training environment reset or learning occurs in this stage.',
        source_candidates=chosen,main_selected=main_selected,source_sha256=dict(queued['source_sha256']),
        input_sha256=dict(queued['input_sha256']),scope=queued['scope'],assessment_rule=queued['assessment_rule'],
        candidate_rule=queued['candidate_rule'],opponent_rule=queued['opponent_rule'],method='conditional_checkpoint_assessment_no_training')
    for key in ('selection_band','selection_n','final_band','final_n','heldout_band','heldout_n'):plan[key]=queued[key]
    paths += [out/'queue_plan.json',out/'development_candidate_review.json']
    for spec in opponents:
        if spec['kind']=='submission':
            paths+=list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]
    for path in paths:plan['input_sha256'][io.relative(path)]=io.sha(path)
    io.verify(plan);io.write(out/'plan.json',plan)
    return plan


def execute(source,out,queued):
    if (source/'failure.json').exists() or not (source/'completion.json').exists():
        raise ValueError('Predecessor has no clean completion')
    completion=io.read(source/'completion.json')
    if not eligible_status(completion['status']):
        io.write(out/'completion.json',dict(status='skipped_main_final_passed',additional_games=0,
            additional_training_steps=0,final_opened=False,heldout_opened=False));return
    plan=build_assessment(source,out,queued,completion)
    if plan is None:
        io.write(out/'completion.json',dict(status='skipped_no_eligible_omitted_snapshot',additional_games=0,
            additional_training_steps=0,final_opened=False,heldout_opened=False));return
    resource=io.resources()
    if resource['commit_headroom_gib']<4.5 or resource['available_memory_gib']<4.:
        raise MemoryError('Insufficient4-worker assessment headroom')
    from tools import league_residual_train as base
    base.torch.set_num_threads(1)
    io.write(out/'runtime.json',dict(pid=os.getpid(),workers=4,phase='checkpoint_assessment',
        started_at=datetime.now(timezone.utc).isoformat(),resources=resource,
        selected_candidates=len(plan['source_candidates']),opponents=len(plan['opponents'])))
    print('Starting conditional checkpoint assessment after clean training completion.',flush=True)
    selected=[dict(seed=r['seed'],selected=dict(spec=r['spec'],rank=r['rank'],step=r['step'])) for r in plan['source_candidates']]
    started=time.perf_counter()
    with io.Sampler() as sample:
        result=base.selection(out,plan,selected)
    result['source_candidates']=result.pop('searches')
    result.update(additional_training_steps=0,assessment_wall_seconds=time.perf_counter()-started,resources=sample.result,
        additional_games=sum(io.read(path)['games'] for stage in ('selection','final','heldout') for path in (out/stage).glob('candidate_*.json')),
        completed_at=datetime.now(timezone.utc).isoformat(),scope=queued['scope'])
    io.verify(plan);io.write(out/'completion.json',result)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--after-run',type=Path,required=True);parser.add_argument('--after-pid',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    try:
        source=args.after_run.resolve();out=args.out.resolve()
        queued=prepare(source,out,args.after_pid);wait_for_source(queued,out);execute(source,out,queued)
    except BaseException:
        io.write(args.out/'failure.json',dict(error=traceback.format_exc(),pid=os.getpid(),at=datetime.now(timezone.utc).isoformat()))
        raise
