"""Reproduce the first league's rounded-health rejudgment using stdlib only.

Optional --source exports the original selected/baseline match records.
Without it, rejudge the portable CSV included alongside the report.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import math

ROOT=Path(__file__).resolve().parents[1]
MODELS=('cem_original','refine_2200')
RULE='user_terminal_health_round4_v2'
FIELDS=('model','opponent','seed','seat','own_health','opp_health','legacy_outcome')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(a,b):
    a,b=float(a),float(b)
    if not all(math.isfinite(x) and 0<=x<=1 for x in (a,b)):raise ValueError('Invalid health')
    a,b=round(a,4),round(b,4)
    return 'win' if a>b else 'loss' if a<b else 'draw'


def run(out,source=None):
    out.mkdir(parents=True,exist_ok=True)
    csv_path=out/'episodes.csv'
    if source:
        rows=[];inputs=[]
        for path in sorted(source.glob('match_*.json')):
            r=json.loads(path.read_text('utf-8'))['result']
            if r['own'] not in MODELS:continue
            inputs.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path)))
            for e in r['episodes']:
                rows.append(dict(model=r['own'],opponent=r['foe'],seed=e['seed'],seat=e['seat'],
                    own_health=e['own_health'],opp_health=e['opp_health'],legacy_outcome=e['outcome']))
        with csv_path.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
        (out/'provenance.json').write_text(json.dumps(dict(
            description='Terminal fields copied from original match records; no replay or retraining.',
            source_files=inputs,csv_sha256=digest(csv_path)),ensure_ascii=False,indent=2)+'\n','utf-8')
    manifest=json.loads((out/'provenance.json').read_text('utf-8'))
    assert digest(csv_path)==manifest['csv_sha256']
    with csv_path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    seen=set();grouped=defaultdict(list)
    for row in rows:
        key=tuple(row[k] for k in ('model','opponent','seed','seat'))
        assert key not in seen;seen.add(key)
        assert row['model'] in MODELS and row['seat'] in ('red','blue')
        row['result']=classify(row['own_health'],row['opp_health'])
        grouped[row['model'],row['opponent']].append(row)
    foes=sorted({r['opponent'] for r in rows});seeds={r['seed'] for r in rows}
    assert len(foes)==16 and len(seeds)==40 and len(rows)==2560
    expected={(s,seat) for s in seeds for seat in ('red','blue')}
    for model in MODELS:
        for foe in foes:assert {(r['seed'],r['seat']) for r in grouped[model,foe]}==expected
    def totals(records):
        c=Counter(r['result'] for r in records);old=Counter(r['legacy_outcome'] for r in records)
        assert set(old)<=set(('kill','died','mutual','timeout'))
        return dict(games=len(records),wins=c['win'],draws=c['draw'],losses=c['loss'],win_rate=c['win']/len(records),
            legacy=dict(wins=old['kill'],draws=old['mutual']+old['timeout'],losses=old['died']))
    models={m:totals([r for r in rows if r['model']==m]) for m in MODELS}
    result=dict(rule=RULE,models=models,per_opponent={f:{m:totals(grouped[m,f]) for m in MODELS} for f in foes},
        direct_match=totals(grouped['refine_2200','cem_original']),csv_sha256=digest(csv_path),
        scope='Rejudgment of consumed historical final games, not new training or an unused test. The originally preselected policy is unchanged. Real student submissions remain untested. Historical CIs and promotion gates are not recalculated.')
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(models,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,default=ROOT/'docs/first_league_health_rejudge_20261009')
    a=p.parse_args();run(a.out.resolve(),a.source.resolve() if a.source else None)
