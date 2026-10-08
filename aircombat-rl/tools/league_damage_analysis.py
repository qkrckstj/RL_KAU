"""Summarize saved damage traces; never run matches or select a policy."""
from argparse import ArgumentParser
from pathlib import Path
from statistics import mean
import os
import time
import traceback

from tools.autolab_cem import read, write, sha
from tools.league_train import verify
from tools.league_validate import process_live


def episode_rows(record, band, n):
    expected = {(seed, seat) for seed in range(band, band + n // 2)
                for seat in ('red', 'blue')}
    if n <= 0 or n % 2:
        raise ValueError('Complete seat pairs required')
    episodes = {(r['seed'], r['seat']): r for r in record['episodes']}
    traces = {(r['seed'], r['seat']): r for r in record['damage_traces']}
    if (set(episodes) != expected or set(traces) != expected
            or len(record['episodes']) != n or len(record['damage_traces']) != n):
        raise ValueError('Incomplete, duplicate or mismatched episode/trace pairs')
    rows = []
    for key in sorted(expected):
        trace = traces[key]
        initial = trace['timeline_1s'][0]['hp']
        terminal = trace['terminal']
        at_30 = trace['snapshots']['30']
        if at_30 is None:
            if terminal['t'] >= 30 - 1e-7:
                raise ValueError('Missing reached 30-second snapshot')
            observed = terminal
        else:
            if not 30 - 1e-7 <= at_30['t'] <= terminal['t'] + 1e-7:
                raise ValueError('Invalid 30-second snapshot time')
            observed = at_30
        damage = initial - observed['hp']
        if damage < -1e-7:
            raise ValueError('Unexpected health increase')
        rows.append(dict(seed=key[0], seat=key[1],
            outcome=episodes[key]['outcome'],
            win=int(episodes[key]['outcome'] == 'kill'),
            loss=int(episodes[key]['outcome'] == 'died'),
            first_hit=trace['first_hit'],
            damage_through_30_or_end=max(0., damage),
            hp_at_30=None if at_30 is None else at_30['hp'],
            ended_before_30=at_30 is None,
            terminal_hp=terminal['hp'], terminal_time=terminal['t'],
            damage_seconds=trace['damage_seconds']))
    return rows


def summarize(rows):
    hits = [r['first_hit'] for r in rows if r['first_hit'] is not None]
    at_30 = [r['hp_at_30'] for r in rows if r['hp_at_30'] is not None]
    early = [r for r in rows if r['ended_before_30']]
    return dict(n=len(rows), wins=sum(r['win'] for r in rows),
        losses=sum(r['loss'] for r in rows),
        draws=sum(not r['win'] and not r['loss'] for r in rows),
        ever_hit=len(hits), first_hit_mean_among_hit=mean(hits) if hits else None,
        hit_by_30=sum(t <= 30 + 1e-7 for t in hits),
        damage_through_30_or_end=mean(r['damage_through_30_or_end'] for r in rows),
        hp_at_30_mean_among_reached=mean(at_30) if at_30 else None,
        reached_30=len(at_30), ended_before_30=len(early),
        early_end_wins=sum(r['win'] for r in early),
        early_end_losses=sum(r['loss'] for r in early),
        terminal_hp=mean(r['terminal_hp'] for r in rows),
        damage_seconds=mean(r['damage_seconds'] for r in rows))


def paired_change(candidate, reference):
    old = {(r['seed'], r['seat']): r for r in reference}
    new = {(r['seed'], r['seat']): r for r in candidate}
    if len(old) != len(reference) or len(new) != len(candidate) or set(old) != set(new):
        raise ValueError('Paired comparison requires identical unique seed/seat keys')
    return {name: mean(new[k][name] - old[k][name] for k in old)
            for name in ('win', 'loss', 'damage_through_30_or_end', 'terminal_hp')}


def run(folder, out, reference):
    completion = read(folder / 'completion.json')
    if completion['status'] != 'diagnostic_complete':
        raise ValueError('Damage diagnostic has not completed')
    if read(folder / 'compatibility.json')['status'] != 'passed':
        raise ValueError('Actual logger compatibility must pass')
    plan = read(folder / 'plan.json')
    verify(plan)
    jobs = [(m['id'], f['id']) for m in plan['models'] for f in plan['opponents']]
    if reference not in {m['id'] for m in plan['models']} or completion['matches'] != len(jobs):
        raise ValueError('Reference or completed schedule mismatch')
    inputs = {str(p.resolve()): sha(p) for p in
              [folder / n for n in ('plan.json', 'completion.json', 'compatibility.json')]}
    rows = {}
    for i, key in enumerate(jobs):
        path = folder / f'match_{i:03d}.json'
        record = read(path)
        if (record['own'], record['foe']) != key:
            raise ValueError('Unexpected model/opponent in trace file')
        rows[key] = episode_rows(record, plan['band'], plan['n'])
        inputs[str(path.resolve())] = sha(path)
    comparisons = [dict(model=model, foe=foe, summary=summarize(rows[model, foe]),
        change_vs_reference=paired_change(rows[model, foe], rows[reference, foe]))
        for model, foe in jobs]
    if any(sha(Path(path)) != digest for path, digest in inputs.items()):
        raise ValueError('Diagnostic input changed during analysis')
    write(out / 'report.json', dict(reference=reference, comparisons=comparisons,
        source_sha256=sha(Path(__file__)), input_sha256=inputs,
        scope='Descriptive open-development damage analysis, with paired seed/seat differences. No promotion or independent generalization claim.',
        interpretation='Damage through 30 or episode end is cumulative observed damage over the shorter interval, not an imputed HP at 30. HP at 30 averages only games reaching that time; use reached counts and early-end WDL to interpret this selected cohort.'))
    write(out / 'completion.json', dict(status='analysis_complete', games=sum(len(v) for v in rows.values()),
        additional_matches=0, model_selection_performed=False))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--after-pid', type=int)
    args = parser.parse_args()
    try:
        if (args.out / 'runtime.json').exists() and process_live(read(args.out / 'runtime.json')['pid']):
            raise ValueError('Analysis already running')
        if (args.out / 'completion.json').exists():
            raise ValueError('Analysis already complete; preserve its output')
        source = sha(Path(__file__))
        write(args.out / 'runtime.json', dict(pid=os.getpid(), waiting_for_pid=args.after_pid,
                                             simulation_workers=0))
        while args.after_pid and process_live(args.after_pid):
            write(args.out / 'progress.json', dict(stage='waiting_for_damage_diagnostic', pid=args.after_pid))
            time.sleep(45)
        if sha(Path(__file__)) != source:
            raise ValueError('Queued analysis source changed')
        run(args.folder.resolve(), args.out.resolve(), args.reference)
        write(args.out / 'progress.json', dict(stage='complete'))
    except Exception:
        write(args.out / 'failure.json', dict(traceback=traceback.format_exc()))
        raise
