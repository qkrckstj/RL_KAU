"""Aiming study progress and measured per-repeat time accounting."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from tools.rear_cpu_reporting import collect, atomic, read


def update(out):
    data=collect(out)
    start=read(out/'started.json')['time']
    if data['training_complete'] and data['evaluation_complete']:
        data['overall_elapsed_seconds']=max((out/f).stat().st_mtime for f in
            ('training_complete.json','evaluation_complete.json'))-start
    from experiments.aim_async.runtime import repeat_elapsed
    for repeat in data['repeats']:
        repeat['learning_and_evaluation_elapsed_seconds']=repeat_elapsed(out,repeat['seed'])
    data['baseline_results']=[read(p) for p in (out/'evaluations').glob('*/completion.json') if read(p)['baseline']]
    lines=['# Predictive aiming CPU learning','',
        f"Original native JSBSim; {data['processes']} physical workers /32 environments. Fixed aiming and HP-exchange reward. All models retained.",'',
        '| Seed | Saved added steps | Training games | Learning seconds | New evaluation games | Evaluation seconds | Wall seconds (overlap included) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in data['repeats']:
        wall='not started' if r['learning_and_evaluation_elapsed_seconds'] is None else f"{r['learning_and_evaluation_elapsed_seconds']:.2f}"
        lines.append(f"| {r['seed']} | {r['additional_steps']:,} | {r['training_games']:,} | {r['training_seconds']:.2f} | {r['evaluation_games']:,} | {r['evaluation_seconds']:.2f} | {wall} |")
    live=data['latest_training_progress']
    if live:lines+=['',f"Live: seed {live['seed']}, {live['additional_steps']:,} added decisions /{live['training_games']:,} completed training games. Saved counters above lag live progress."]
    lines+=['',f"New completed evaluations: {data['new_evaluation_games']}; reused parent games: {data['reused_baseline_games']}.",
        f"Elapsed wall time: {data['overall_elapsed_seconds']:.2f}s; completed evaluation process time: {data['evaluation_seconds']:.2f}s. Training and evaluation overlap.",'',
        '| Model /panel | Added steps | Wins /games | Win rate | Gain vs same-roster parent | Dealt /received HP | First-shot rate | Connected firing seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    rows=[dict(seed='parent',kind=e['kind'],steps=0,summary=e['summary'],win_gain_pp=None) for e in data['baseline_results']]+data['evaluation_results']
    for e in rows:
        s=e['summary'];gain='baseline' if e['win_gain_pp'] is None else f"{e['win_gain_pp']:+.2f}pp"
        first='not recorded' if 'first_shot_advantage_rate' not in s else f"{s['first_shot_advantage_rate']*100:.2f}%"
        fire='not recorded' if 'connected_fire_seconds' not in s else f"{s['connected_fire_seconds']:.3f}"
        lines.append(f"| {e['seed']} /{e['kind']} | {e['steps']:,} | {s['wins']} /{s['games']} | {s['win_rate']*100:.2f}% | {gain} | {s['damage_dealt']:.4f} /{s['damage_received']:.4f} | {first} | {fire} |")
    if data['pending_jobs']:
        lines+=['','Pending evaluations (partial games are separate from completed-job totals):']
        lines += [f"- {j['checkpoint']} /{j['kind']}: {j['completed_cells']} cells; blocked={j['blocked']}." for j in data['pending_jobs']]
    lines+=['','Roster is frozen within the study. Similar historical opponents have one representative; full artifacts and small-gain candidates remain archived.',
        'Consumed development panels support adaptive selection, not unseen final validation. No automatic publication. Rear scores do not gate continuation.','']
    atomic(out/'PROGRESS.json',json.dumps(data,indent=2)+'\n')
    text='\n'.join(lines)
    atomic(out/'PROGRESS.md',text);atomic(out/'SUMMARY.md',text)
    atomic(out/'timing_ledger.json',json.dumps(dict(repeats=data['repeats'],new_evaluation_games=data['new_evaluation_games'],
        reused_baseline_games=data['reused_baseline_games'],elapsed_seconds=data['overall_elapsed_seconds'],
        evaluation_seconds=data['evaluation_seconds'],stage_times_overlap=True),indent=2)+'\n')
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
    out=a.out.resolve()
    with (out/'reporter.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic(out/'reporter_runtime.json',json.dumps(dict(pid=os.getpid(),status='running'))+'\n')
        while True:
            data=update(out)
            if not a.watch or data['training_failed'] or (data['training_complete'] and data['evaluation_complete']):break
            time.sleep(30)
        atomic(out/'reporter_runtime.json',json.dumps(dict(pid=os.getpid(),status='stopped'))+'\n')
