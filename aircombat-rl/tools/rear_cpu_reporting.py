"""Read-only reporting for native CPU learning and asynchronous evaluations."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def collect(out, now=None):
    plan = read(out/'plan.json')
    evaluations = [read(p) for p in (out/'evaluations').glob('*/completion.json')]
    repeats = []
    for seed in plan['seeds']:
        path = out/f's{seed}_result.json'
        training = read(path) if path.exists() else {}
        own = [e for e in evaluations if e['seed'] == seed and not e['baseline']]
        repeats.append(dict(seed=seed,
            additional_steps=training.get('additional_steps', 0),
            training_games=training.get('training_games', 0),
            training_seconds=training.get('training_seconds', 0),
            evaluation_games=sum(e['actual_new_games'] for e in own),
            evaluation_seconds=sum(e['wall_seconds'] for e in own)))
    pending = []
    for p in (out/'queue').glob('*.json'):
        job = read(p)
        folder = out/'evaluations'/job['job_id']
        if not (folder/'completion.json').exists():
            pending.append(dict(checkpoint=job['checkpoint_id'], kind=job['kind'],
                completed_cells=len(list((folder/'cells').glob('*.json'))),
                blocked=(folder/'blocked.json').exists()))
    diagnostics = [read(p) for p in (out/'diagnostics').glob('*/summary.json')]
    results = [dict(seed=e['seed'], kind=e['kind'], steps=e['additional_steps'],
        summary=e['summary'], win_gain_pp=None if e.get('win_gain') is None else 100*e['win_gain'])
        for e in evaluations if not e['baseline']]
    results.sort(key=lambda e: (e['seed'], e['steps'], e['kind']))
    progress_file = out/'training_progress.json'
    return dict(recorded_unix=time.time() if now is None else now,
        repeats=repeats, processes=plan['processes'],
        latest_training_progress=read(progress_file) if progress_file.exists() else {},
        new_evaluation_games=sum(e['actual_new_games'] for e in evaluations),
        reused_baseline_games=sum(e['summary']['games'] for e in evaluations if e.get('reused_from_run')),
        evaluation_seconds=sum(e['wall_seconds'] for e in evaluations),
        diagnostic_games=sum(d.get('diagnostic_games', 0) for d in diagnostics),
        diagnostic_seconds=sum(d.get('wall_seconds', 0) for d in diagnostics),
        pending_jobs=pending, evaluation_results=results,
        monitor_results=[e for e in results if e['kind'] == 'monitor'],
        overall_elapsed_seconds=(time.time() if now is None else now)-read(out/'started.json')['time'],
        training_complete=(out/'training_complete.json').exists(),
        evaluation_complete=(out/'evaluation_complete.json').exists(),
        training_failed=(out/'training_failure.json').exists())


def render(data):
    lines = ['# CPU rear automatic learning progress', '',
        f"Original native JSBSim; {data['processes']} CPU workers / 32 environments. All checkpoints remain selection candidates, including small gains.", '',
        '| Repeat | Saved additional steps | Completed training games | Learning seconds | New completed evaluation games | Evaluation seconds |',
        '|---|---:|---:|---:|---:|---:|']
    for r in data['repeats']:
        lines.append(f"| {r['seed']} | {r['additional_steps']:,} | {r['training_games']:,} | {r['training_seconds']:.2f} | {r['evaluation_games']:,} | {r['evaluation_seconds']:.2f} |")
    p = data['latest_training_progress']
    if p:
        lines += ['', f"Latest live training sample: seed {p['seed']}, {p['additional_steps']:,} additional steps, {p['training_games']:,} completed games. The table uses completed checkpoint counters and their measured learning time."]
    lines += ['',
        f"New completed policy evaluation games: **{data['new_evaluation_games']:,}**. Reused unchanged parent evaluation games: **{data['reused_baseline_games']:,}**.",
        f"Diagnostic games: **{data['diagnostic_games']}** ({data['diagnostic_seconds']:.2f}s), replayed consumed conditions; not new validation conditions.",
        f"Overall elapsed: **{data['overall_elapsed_seconds']:.2f}s**. Completed evaluation process time: {data['evaluation_seconds']:.2f}s. Stage times overlap; do not sum them as total wall time.", '',
        '| Seed / fixed panel | Added training steps | Games | Win rate | Gain vs parent | Actual rear damage | Clean rear wins | Rear streak seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for e in data['evaluation_results']:
        s = e['summary']
        gain = 'unavailable' if e['win_gain_pp'] is None else f"{e['win_gain_pp']:+.2f}pp"
        lines.append(f"| {e['seed']} / {e['kind']} | {e['steps']:,} | {s['games']} | {100*s['win_rate']:.2f}% | {gain} | {s['rear_damage']:.5f} | {s.get('clean_rear_wins', 'n/a')} | {s.get('max_rear_wez_streak', 0):.3f} |")
    if data['pending_jobs']:
        lines += ['', 'Pending evaluation cells are recorded separately from completed-job totals:']
        for job in data['pending_jobs']:
            lines.append(f"- {job['checkpoint']} / {job['kind']}: {job['completed_cells']} cached cells; blocked={job['blocked']}.")
    lines += ['', 'Fixed development panels support candidate selection and learning curves; they are not unseen final tests. No automatic frontline promotion or GitHub publication.', '']
    return '\n'.join(lines)


def atomic(path, text):
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def update(out):
    data = collect(out)
    atomic(out/'PROGRESS.json', json.dumps(data, indent=2)+'\n')
    atomic(out/'PROGRESS.md', render(data))
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--watch', action='store_true')
    a = p.parse_args()
    out = a.out.resolve()
    with (out/'reporter.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(out/'reporter_runtime.json', json.dumps(dict(pid=os.getpid(), status='running'))+'\n')
        while True:
            data = update(out)
            if not a.watch or data['training_failed'] or (data['training_complete'] and data['evaluation_complete']):
                break
            time.sleep(30)
        atomic(out/'reporter_runtime.json', json.dumps(dict(pid=os.getpid(), status='stopped'))+'\n')
