"""Read-only historical audit under user-specified terminal health ordering.

Round both health values to four decimals; equal values draw by user instruction.
No training, simulator changes, or Git operations.
Per-file counts are record inventories, not independent experimental evidence.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verdict(row):
    own, foe = float(row['own_health']), float(row['opp_health'])
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (own, foe)):
        return 'invalid'
    own, foe = round(own, 4), round(foe, 4)
    if own > foe:
        return 'win'
    if own < foe:
        return 'loss'
    return 'draw'


def rows_in(value):
    if isinstance(value, dict):
        if all(k in value for k in ('own_health', 'opp_health', 'outcome')) and ('seed' in value or 'steps' in value):
            yield value
            return
        for item in value.values():
            yield from rows_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from rows_in(item)


def counts(rows):
    c = Counter()
    for row in rows:
        c['records'] += 1
        old = 'win' if row['outcome'] == 'kill' else 'loss' if row['outcome'] == 'died' else 'draw'
        new = verdict(row)
        c['old_' + old] += 1
        c[new] += 1
        c[old + '_to_' + new] += 1
    return dict(c)


def audit(destination):
    destination.mkdir(parents=True, exist_ok=False)
    runs = ROOT / 'aircombat-rl/runs'
    totals = Counter(); scanned = 0; errors = []
    fields = ['source', 'sha256', 'records', 'old_win', 'old_draw', 'old_loss', 'win', 'loss', 'draw', 'invalid']
    with (destination/'file_inventory.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for parent, dirs, files in os.walk(runs):
            dirs[:] = sorted(d for d in dirs if d not in ('source_snapshot', '__pycache__', '.git', 'checkout') and not d.startswith('replay_snapshot'))
            for name in sorted(files):
                if not name.endswith('.json'):
                    continue
                path = Path(parent)/name; scanned += 1
                try:
                    raw = path.read_bytes(); obj = json.loads(raw)
                    rows = list(rows_in(obj))
                    if rows:
                        c = counts(rows); totals.update(c); totals['files_with_records'] += 1
                        writer.writerow(dict(source=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(raw).hexdigest(), **{k:c.get(k, 0) for k in fields[2:]}))
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    errors.append(dict(source=path.relative_to(ROOT).as_posix(), error=str(exc)))
                if scanned % 10000 == 0:
                    print(json.dumps(dict(scanned=scanned, recorded_files=totals['files_with_records'])), flush=True)
    # Latest comparable screens: use only canonical match files, deduplicate
    # identical jobs, and retain policy/opponent/IC/seat/action-RNG identities.
    panels = {}
    for run in ('league_compact_inactive_assess_20261008', 'league_compact_rotated_assess_20261008'):
        grouped = defaultdict(list); seen = {}; conflicts = []; sources = []
        for path in sorted((runs/run/'matches').glob('*.json')):
            raw = path.read_bytes(); data = json.loads(raw); job = data['job']
            sources.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(raw).hexdigest()))
            for row in data['result']['episodes']:
                identity = json.dumps([job['own'], job['foe'], row['seed'], row['seat']], sort_keys=True)
                fingerprint = json.dumps(row, sort_keys=True)
                if identity in seen:
                    if seen[identity] != fingerprint: conflicts.append(identity)
                    continue
                seen[identity] = fingerprint
                grouped[job['own']['id']].append(row)
        stats = {}
        for policy, rows in grouped.items():
            c = counts(rows); n = c['records']
            c['win_rate'] = c.get('win', 0)/n
            stats[policy] = c
        panels[run] = dict(policies=stats, identity_conflicts=conflicts, sources=sources)
    result = dict(rule='user_terminal_health_round4_v2', rule_authority='User correction; not asserted to be verified upstream tournament rules',
        rule_text='At existing terminal boundary: round both health values to four decimals, > win, < loss, equal draw.',
        scanned_json_files=scanned, inventory_totals=dict(totals), errors=errors, panels=panels,
        scope='All readable local runs JSON except source/replay snapshots. Inventory may include repeated training/development/test/smoke records: NEVER interpret its totals as pooled win rate or independent games. Only panels use canonical match identities.',
        limitation='Equal rounded values are draws by user instruction. These are historical rescores, not new independent tests. Non-JSON/missing/outside-runs records are not covered.',
        training_started=False, historical_files_modified=False)
    (destination/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('panels','errors')}, ensure_ascii=False), flush=True)
    print(json.dumps({'panels':{k:v['policies'] for k,v in panels.items()},'errors':len(errors)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out.resolve())
