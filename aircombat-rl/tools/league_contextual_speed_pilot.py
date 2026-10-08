"""Conditional queued development pilot after a completed unsuccessful search.

Stdlib-only while waiting. A native process handle prevents PID reuse from
starting overlapping simulations. This pilot does not open a final test,
promote a policy, or publish to GitHub.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import ctypes
from ctypes import wintypes
import multiprocessing as mp
import os
import shutil
import sys
import traceback
from tools.league_thread_benchmark import (
    ROOT, read, write, sha, verify, resources, environment, process_live, THREAD_KEYS,
)

CONTEXT_GRID=((5.,1000.,500.,1.),(15.,1000.,500.,1.),(25.,1000.,500.,1.),
              (5.,1000.,575.,1.),(15.,1000.,575.,1.),(25.,1000.,575.,1.),
              (5.,0.,550.,1.5),(25.,2000.,575.,.5))


def relative(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def eligible_status(status):
    if status=='repair_profile_passed':return False
    if status in ('no_development_improvement','repair_profile_failed'):return True
    raise ValueError('Unexpected predecessor outcome; inspect rather than retry')


def comparable(result):
    # Different model IDs are intentional for the disabled identity control.
    return {k:result[k] for k in ('foe','band','summary','episodes')}


def queue_plan(source,out,pid):
    if out.exists():raise FileExistsError('Use a new queued pilot folder')
    old=read(source/'plan.json');verify(old)
    runtime=read(source/'runtime.json')
    if runtime['pid']!=pid:raise ValueError('Wrong predecessor PID')
    if any(os.environ.get(k)!='1' for k in THREAD_KEYS):raise ValueError('Set thread limits before Python')
    sources=dict(old['source_sha256'])
    for name in ('tools/league_contextual_speed_pilot.py','experiments/league/contextual_speed.py',
                 'tools/league_contextual_speed_export.py'):
        sources[name]=sha(ROOT/name)
    inputs=dict(old['input_sha256']);inputs[relative(source/'plan.json')]=sha(source/'plan.json')
    plan=dict(source=relative(source),after_pid=pid,source_started_at=runtime['started_at'],
        environment=environment(),grid=CONTEXT_GRID,screen_band=62000000,confirmation_band=62010000,
        compatibility_band=170000000,screen_n=8,confirmation_n=24,confirm_top=2,
        source_sha256=sources,input_sha256=inputs,
        eligibility='Only after completed no_development_improvement or repair_profile_failed. Skip on repair_profile_passed; crash/incomplete predecessor requires inspection.',
        scope='Development-only coefficient grid with fresh screening/confirmation ICs. Consumed temporal opponents are development foes. No final test, untrained-opponent claim, promotion or publication.')
    write(out/'queue_plan.json',plan)
    for band in (plan['screen_band'],plan['confirmation_band']):
        path=ROOT/f'runs/context_pilot_reservation_{band}.json'
        reservation=dict(out=relative(out),band=band)
        if path.exists() and read(path)!=reservation:raise ValueError('Development band reserved elsewhere')
        write(path,reservation)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    return plan


def wait_for_source(source,out,pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x100000|0x1000,False,pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        times=[wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle,*(ctypes.byref(v) for v in times)):raise ctypes.WinError(ctypes.get_last_error())
        created=((times[0].dwHighDateTime<<32)+times[0].dwLowDateTime)/1e7-11644473600
        expected=datetime.fromisoformat(read(source/'runtime.json')['started_at']).timestamp()
        if abs(created-expected)>10:raise ValueError('PID creation time differs from predecessor runtime')
        plan=queue_plan(source,out,pid)
        write(out/'queue_runtime.json',dict(pid=os.getpid(),after_pid=pid,source_process_created_unix=created,
            started_at=datetime.now(timezone.utc).isoformat(),simulation_workers_while_waiting=0,
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules))
        while True:
            state=kernel.WaitForSingleObject(handle,45000)
            if state==0:break
            if state!=258:raise ctypes.WinError(ctypes.get_last_error())
        verify(plan)
        if environment()!=plan['environment']:raise ValueError('Environment changed while queued')
        if not (source/'completion.json').exists():raise ValueError('Predecessor exited without completion')
        return plan
    finally:kernel.CloseHandle(handle)


def build_plan(source,out,queued):
    from tools.league_repair_train import configuration_of
    from tools.league_train import unique_entrants
    from tools.league_tournament_metrics import code_groups,group_weights
    from tools.league_contextual_speed_export import export
    from tools.league_numpy_matches import code_signature
    from tools.league_symmetric_train import immutable_json
    old=read(source/'plan.json');verify(old)
    completed=read(source/'completion.json')
    if not eligible_status(completed['status']):return None
    for seed in old['seeds']:
        if read(source/f's{seed}/result.json')['status']!='complete':raise ValueError('Incomplete independent search')
    warm=old['warm_start'];config=configuration_of(warm)
    dev_paths=sorted(source.glob('s*/g*/development/candidate_000.json'))
    additions=[read(p)['request']['spec'] for p in dev_paths]
    opponents=unique_entrants(old['opponents']+additions)
    if len({s['id'] for s in opponents})!=len(opponents):raise ValueError('Duplicate opponent ID')
    disabled=export(out/'models/disabled',dict(**config,context=[.5,0.,300.,0.]),out.name+'_disabled')
    candidates=[export(out/f'models/c{i}',dict(**config,context=list(values)),f'{out.name}_c{i}')
                for i,values in enumerate(queued['grid'])]
    signatures={code_signature(s) for s in [disabled]+candidates}
    if len(signatures)!=1:raise ValueError('Candidate source mismatch')
    signature=signatures.pop()
    groups=code_groups(opponents);gw=group_weights(groups)
    baseline=read(source/'baseline/candidate_000.json')
    weakness={f['foe']:1-f['summary']['score']+.1 for f in baseline['results'] if f['foe'] in old['temporal_ids']}
    weights=[.4/len(opponents)+.3*gw[s['id']]+.3*weakness.get(s['id'],0)/sum(weakness.values()) for s in opponents]
    inputs=dict(queued['input_sha256'])
    paths=[source/'completion.json',source/'frozen_selection.json',source/'baseline/candidate_000.json']+dev_paths
    paths += [source/f's{s}/result.json' for s in old['seeds']]
    for spec in opponents+[warm,disabled]+candidates:
        if spec['kind']=='submission':
            folder=ROOT/spec['design'];paths+=list(folder.glob('*.py'))+[ROOT/spec['weights']]
            paths += [folder/n for n in ('policy_net.json','entrant.json','artifact_sha256.json') if (folder/n).exists()]
    for path in paths:inputs[relative(path)]=sha(path)
    plan=dict(queued,warm_start=warm,disabled=disabled,candidates=candidates,opponents=opponents,
        groups=groups,weights=weights,temporal_ids=old['temporal_ids'],execution=old['execution'],
        provisional_context_signature=signature,input_sha256=inputs,
        qualification_rule='Before any split-pool pilot jobs: actual disabled-policy identity vs warm, and original-vs-NumPy whole-flight parity for every active context configuration. No provisional signature admitted on failure.',
        selection_rule='Retain two highest frozen weighted CEM-objective-plus-healthy-win candidates from screen; compare to warm on fresh confirmation. Report repair criteria; no final model promotion.')
    verify(plan);immutable_json(out/'plan.json',plan)
    return plan


def qualify(plan,out):
    from tools.league_matches import duel
    from tools.league_numpy_matches import duel as numpy_duel
    from tools.league_symmetric_train import immutable_json
    foes={s['id']:s for s in plan['opponents']}
    names=('ace','evader','circler','refine_2200','temporal_extend_right','temporal_weave_right')
    approved=plan['execution']['approved_signatures']+[plan['provisional_context_signature']]
    records=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        for name in names:
            args=(foes[name],plan['compatibility_band'],2)
            original=duel(plan['warm_start'],*args)
            disabled=duel(plan['disabled'],*args)
            if comparable(original)!=comparable(disabled):raise ValueError('Disabled context changed full flight')
            pure=pool.submit(numpy_duel,plan['disabled'],*args,approved).result()
            if comparable(disabled)!=comparable(pure):raise ValueError('Disabled loader mismatch')
            records.append(dict(kind='disabled_identity_and_loader',original=original,disabled=disabled,pure=pure))
        for model in plan['candidates']:
            for name in ('ace','temporal_extend_right'):
                args=(foes[name],plan['compatibility_band'],2)
                original=duel(model,*args)
                pure=pool.submit(numpy_duel,model,*args,approved).result()
                if comparable(original)!=comparable(pure):raise ValueError('Active context loader mismatch')
                records.append(dict(kind='active_loader',original=original,pure=pure))
    result=dict(status='whole_flight_compatibility_passed',approved_context_signature=plan['provisional_context_signature'],
        games=100,records=records,scope='Reused compatibility ICs; mechanics/loader identity only. No independent performance evidence.')
    immutable_json(out/'compatibility.json',result)
    return result


def run(source,out,queued):
    from tools.league_repair_train import Evaluation
    from tools.league_temporal_repair_train import rank,decision
    from tools.league_tournament_metrics import profile
    from tools.league_split_pool import SplitPool
    from tools.league_symmetric_train import immutable_json
    completed=read(source/'completion.json')
    if not eligible_status(completed['status']):
        write(out/'completion.json',dict(status='not_eligible_prior_passed',source_status=completed['status'],games=0));return
    before=resources()
    if before['commit_headroom_gib']<5.5 or before['available_memory_gib']<4:
        raise ValueError('Insufficient commit or physical headroom after predecessor exit')
    plan=build_plan(source,out,queued)
    write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),workers=16,resources=before))
    compatibility=qualify(plan,out)
    if compatibility['status']!='whole_flight_compatibility_passed':raise ValueError('Missing compatibility')
    e=plan['execution'];approved=e['approved_signatures']+[compatibility['approved_context_signature']]
    evaluate=Evaluation()
    with SplitPool(e['workers'],e['neural_workers'],approved,e['pure_warmup']+[plan['disabled']],e['neural_warmup']) as pool:
        immutable_json(out/'initialized_workers.json',pool.initialized_workers)
        write(out/'progress.json',dict(stage='context_screen'))
        screen=evaluate(pool,[plan['warm_start']]+plan['candidates'],plan['opponents'],plan['screen_band'],
            plan['screen_n'],out/'screen',plan['weights'])
        order=sorted(range(len(plan['candidates'])),key=lambda i:rank(screen[i+1],plan['weights']),reverse=True)
        chosen=[plan['candidates'][i] for i in order[:plan['confirm_top']]]
        immutable_json(out/'frozen_confirmation.json',dict(candidates=chosen,indices=order[:plan['confirm_top']],
            selection_used_screen_only=True,plan_sha256=sha(out/'plan.json')))
        write(out/'progress.json',dict(stage='context_confirmation'))
        confirm=evaluate(pool,[plan['warm_start']]+chosen,plan['opponents'],plan['confirmation_band'],
            plan['confirmation_n'],out/'confirmation')
        comparisons=[decision(r,confirm[0],plan) for r in confirm[1:]]
        best=max(range(len(chosen)),key=lambda i:rank(confirm[i+1],plan['weights']))
        result=dict(status='development_pilot_complete',candidates=chosen,comparisons=comparisons,
            best_candidate=chosen[best],best_repairs_profile=comparisons[best]['repair_passed'],
            best_beats_baseline_training_rank=rank(confirm[best+1],plan['weights'])>rank(confirm[0],plan['weights']),
            retained_policy=plan['warm_start'],
            profiles=[dict(model=r['request']['spec']['id'],profile=profile(r['results'],plan['groups'])) for r in confirm],
            screen_games=sum(r['metrics']['games'] for r in screen),confirmation_games=sum(r['metrics']['games'] for r in confirm),
            final_opened=False,policy_promoted=False,scope=plan['scope'],
            next='Analyze pilot evidence before independent CEM context optimization or another structural change. Preserve prior policy; pilot is not a final test.')
        verify(plan);immutable_json(out/'completion.json',result)
        write(out/'progress.json',dict(stage='complete'))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('runs/league_temporal_repair_20261007'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--after-pid',type=int,required=True)
    args=parser.parse_args()
    try:
        source,out=args.source.resolve(),args.out.resolve()
        queued=wait_for_source(source,out,args.after_pid)
        run(source,out,queued)
    except BaseException:
        write(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
