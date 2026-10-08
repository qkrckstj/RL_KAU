"""Lightweight waiter, then independently recompute a completed final report.

The waiter imports only stdlib and the stdlib-only resource helper. It never
reads active progress JSON, starts matches, promotes policies or publishes.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import ctypes
from ctypes import wintypes
import json
import os
import sys
import time
import traceback
from tools.league_thread_benchmark import ROOT, read, sha, write, verify, resources


def relative(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def collect_report(source, out):
    # NumPy and policy/evaluation helpers are imported only after simulations end.
    from tools.league_tournament_metrics import profile, compare, paired_gain
    from tools.league_cached_evaluation import validate
    from tools.league_damage_analysis import episode_rows, summarize
    from tools.league_matches import summary
    plan=read(source/'plan.json');verify(plan)
    completed=read(source/'completion.json');selection=read(source/'frozen_selection.json')
    if completed['status'] not in ('tournament_profile_passed','tournament_profile_failed'):
        raise ValueError('Expected a completed final evaluation')
    if completed['selected']!=selection['chosen'] or not completed['selected_before_test']:
        raise ValueError('Final result differs from pre-test selection')
    schedule=read(source/'final/schedule.json')
    if not schedule['traced']: raise ValueError('Expected final damage observations')
    rows=[];slim=[];input_paths=[source/n for n in ('plan.json','completion.json','frozen_selection.json','final_decision.json')]
    input_paths.append(source/'final/schedule.json')
    for i,request in enumerate(schedule['requests']):
        path=source/f'final/candidate_{i:03d}.json';r=read(path);validate(r,request);input_paths.append(path)
        if request['band']!=plan['final_band'] or request['n']!=plan['final_n'] or request['opponents']!=plan['opponents']:
            raise ValueError('Final request conditions differ from plan')
        all_damage=[];per_foe=[]
        for f in r['results']:
            if summary(f['episodes'])!=f['summary']: raise ValueError('Episode summary does not reproduce')
            damage=episode_rows(f,request['band'],request['n']);all_damage.extend(damage)
            per_foe.append(dict(foe=f['foe'],wdl={k:f['summary'][k] for k in ('wins','draws','losses')},
                win_rate=f['summary']['rate'],score=f['summary']['score'],damage=summarize(damage)))
        rows.append(dict(spec=request['spec'],profile=profile(r['results'],plan['groups']),
            wdl={k:sum(f['summary'][k] for f in r['results']) for k in ('wins','draws','losses')},
            damage=summarize(all_damage),per_opponent=per_foe))
        slim.append(dict(request=request,results=[{k:v for k,v in f.items() if k!='damage_traces'} for f in r['results']]))
    if [r['spec'] for r in rows]!=plan['final_models'] or rows[-1]['spec']!=selection['chosen']:
        raise ValueError('Wrong evaluated models')
    comparisons=[compare(slim[-1]['results'],r['results'],plan['groups'],final=True) for r in slim[:2]]
    warm=compare(slim[-1]['results'],slim[2]['results'],plan['groups'],final=True)
    if comparisons!=completed['comparisons'] or warm!=completed['warm_comparison']:
        raise ValueError('Final summary does not reproduce from raw records')
    passed=all(c['profile_passed'] for c in comparisons)
    if passed!=(completed['status']=='tournament_profile_passed'): raise ValueError('Inconsistent profile decision')
    result=dict(status='analysis_complete',profile_passed=passed,selected=selection['chosen'],
        selected_before_test=True,records=rows,comparisons=comparisons,warm_comparison=warm,
        temporal_opened=completed['temporal_opened'],promotion_performed=False,
        scope='Raw completed records rechecked; no simulations or model selection. IC confidence intervals condition on the fixed learned policies and opponent panel. Code groups are sensitivity analysis, not actual tournament probabilities.')
    if completed['temporal_opened']:
        panel=plan['temporal'];temporal=[];raw=[]
        audit=read(source/'temporal/completion.json');input_paths.append(source/'temporal/completion.json')
        if audit['selected']!=selection['chosen'] or audit['model_selection_performed']:
            raise ValueError('Temporal audit changed the chosen model')
        schedule2=read(source/'temporal/schedule.json');input_paths.append(source/'temporal/schedule.json')
        for i,request in enumerate(schedule2['requests']):
            path=source/f'temporal/candidate_{i:03d}.json';r=read(path);validate(r,request);input_paths.append(path)
            if request['band']!=panel['band'] or request['n']!=panel['n'] or request['opponents']!=panel['opponents']:
                raise ValueError('Temporal request differs from frozen panel')
            raw.append(r)
            if any(summary(f['episodes'])!=f['summary'] for f in r['results']):
                raise ValueError('Temporal episode summary does not reproduce')
            temporal.append(dict(spec=request['spec'],wdl={k:sum(f['summary'][k] for f in r['results']) for k in ('wins','draws','losses')},
                per_opponent=[dict(foe=f['foe'],summary=f['summary']) for f in r['results']]))
        if temporal[-1]['spec']!=selection['chosen']: raise ValueError('Wrong temporal chosen model')
        weights={f['id']:1/len(panel['opponents']) for f in panel['opponents']}
        gains=[dict(reference=r['request']['spec']['id'],paired_gain=paired_gain(raw[-1]['results'],r['results'],weights)) for r in raw[:-1]]
        if gains!=audit['comparisons']: raise ValueError('Temporal summary differs from raw records')
        result['temporal']=dict(records=temporal,comparisons=gains,scope=panel['limits'])
    inputs={relative(p):sha(p) for p in input_paths}
    result['input_sha256']=inputs
    result['created_at']=datetime.now(timezone.utc).isoformat()
    write(out/'report.json',result)
    lines=['# Fixed-choice final evaluation', '', f"Final profile passed: **{passed}**. Selected: `{selection['chosen']['id']}`.",
        '', 'No policy promotion or GitHub publication is performed by this report.', '',
        '| Model | Wins | Draws | Losses | Uniform win rate | Group-balanced win rate | Lower-quarter score | Worst score | Losing matchups |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m=row['profile'];w=row['wdl'];lines.append(f"| {row['spec']['id']} | {w['wins']} | {w['draws']} | {w['losses']} | {m['mean_win_rate']:.4f} | {m['group_balanced_win_rate']:.4f} | {m['lower_quarter_score']:.4f} | {m['worst_score']:.4f} | {m['losing_matchups']} |")
    lines+=['','## Paired changes','']
    for ref,c in zip(rows[:3],comparisons+[warm]):
        lines.append(f"- Versus {ref['spec']['id']}: uniform gain {c['uniform_paired_gain']['mean']:.4f}, IC bootstrap 95% CI {c['uniform_paired_gain']['ci95']}; group gain {c['group_paired_gain']['mean']:.4f}, CI {c['group_paired_gain']['ci95']}; legacy safeguards passed {c['legacy_safeguards_passed']}.")
    lines+=['','## Damage and limits','','Damage through 30 seconds uses terminal damage if an episode ended earlier; it does not extrapolate health to a later time. First-hit means include only episodes in which a hit occurred. These are descriptive outcomes, not extra selection criteria.','',
        '| Model | Hit by 30s | Mean damage through 30s or end | Mean terminal HP |', '|---|---:|---:|---:|']
    for row in rows:
        d=row['damage'];lines.append(f"| {row['spec']['id']} | {d['hit_by_30']}/{d['n']} | {d['damage_through_30_or_end']:.4f} | {d['terminal_hp']:.4f} |")
    lines+=['',f"Temporal panel opened: {result['temporal_opened']}. Its completion alone does not establish superiority.",'',result['scope']]
    if 'temporal' in result:
        lines+=['','## Untrained temporal opponents','']
        for r in result['temporal']['records']: lines.append(f"- {r['spec']['id']}: {r['wdl']}")
        for c in result['temporal']['comparisons']: lines.append(f"- Selected vs {c['reference']}: {c['paired_gain']}")
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if any(sha(ROOT/name)!=digest for name,digest in inputs.items()): raise ValueError('Input changed during report')
    verify(plan);write(out/'completion.json',dict(status='analysis_complete',profile_passed=passed,temporal_opened=result['temporal_opened']))
    return result


def watch(source,out,pid):
    if read(source/'runtime.json')['pid']!=pid: raise ValueError('Wait PID does not match source runtime')
    if out.exists(): raise FileExistsError('Use new report folder for watcher')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x100000,False,pid)
    if not handle: raise ctypes.WinError(ctypes.get_last_error())
    frozen=dict(source=relative(source),source_plan_sha256=sha(source/'plan.json'),report_source_sha256=sha(__file__),
        controller_pid=pid,watcher_pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
        scope='45-second sampled resources from watcher launch, not whole-run peak; no matches while waiting. Native process handle prevents PID-reuse ambiguity.')
    write(out/'watch_plan.json',frozen)
    try:
        with (out/'resource_samples.jsonl').open('a',encoding='utf-8') as log:
            while True:
                log.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),**resources()))+'\n');log.flush()
                state=kernel.WaitForSingleObject(handle,45000)
                if state==0:break
                if state!=258:raise ctypes.WinError(ctypes.get_last_error())
    finally:kernel.CloseHandle(handle)
    if sha(__file__)!=frozen['report_source_sha256'] or sha(source/'plan.json')!=frozen['source_plan_sha256']:
        raise ValueError('Queued report inputs changed')
    if not (source/'completion.json').exists():
        write(out/'completion.json',dict(status='source_incomplete',report_generated=False));return
    collect_report(source,out)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--after-pid',type=int)
    args=parser.parse_args()
    try:
        if args.after_pid:watch(args.source.resolve(),args.out.resolve(),args.after_pid)
        else:collect_report(args.source.resolve(),args.out.resolve())
    except Exception:write(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
