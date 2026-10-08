"""Recombine learned opening and post-probe parent on consumed development ICs."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import traceback
from tools.autolab_cem import ROOT, read, sha
from tools.league_reliable_io import write
from tools.league_symmetric_train import immutable_json
from tools.league_train import verify
from tools.league_repair_train import Evaluation, configuration_of
from tools.league_matches import duel, initialize_worker
from tools.league_separated_export import export
from tools.league_thread_benchmark import process_live, resources, environment, THREAD_KEYS
from tools.league_tournament_metrics import profile, compare, ranking


def relative(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def freeze(out, source, workers):
    if read(source/'completion.json')['status']!='no_profile_selection': raise ValueError('Expected completed failed selection')
    if (source/'runtime.json').exists() and process_live(read(source/'runtime.json')['pid']): raise ValueError('Prior assessment still live')
    if any(os.environ.get(k)!='1' for k in THREAD_KEYS): raise ValueError('Set thread limits before Python')
    if (out/'plan.json').exists():
        p=read(out/'plan.json');verify(p)
        if (p['source'],p['workers'],p['environment'])!=(relative(source),workers,environment()): raise ValueError('Changed probe settings')
        return p
    old=read(source/'plan.json');verify(old)
    records=[read(source/f'selection/candidate_{i:03d}.json') for i in range(2+len(old['candidates']))]
    warm=max(records[2:],key=lambda r:ranking(profile(r['results'],old['groups'])))
    config=configuration_of(warm['request']['spec'])
    alternative=next(r for r in records[2:] if r['request']['spec']!=warm['request']['spec'])
    base=dict(probe=config['parent'],parent=config['parent'],interceptor=config['interceptor'])
    control=export(out/'models/identity',base,out.name+'_identity')
    parameters=[old['parent']['parameters'],configuration_of(alternative['request']['spec'])['parent']]
    models=[export(out/f'models/parent_{i}',dict(base,parent=p),out.name+f'_parent_{i}') for i,p in enumerate(parameters)]
    sources=dict(old['source_sha256'])
    for name in ('experiments/league/separated_probe.py','tools/league_separated_export.py','tools/league_separated_probe.py'):
        sources[name]=sha(ROOT/name)
    inputs=dict(old['input_sha256'])
    extra=[source/'plan.json',source/'completion.json']+[source/f'selection/candidate_{i:03d}.json' for i in range(len(records))]
    for spec in [control]+models:
        extra.extend((ROOT/spec['design']).glob('*'))
    for path in extra:
        if path.is_file(): inputs[relative(path)]=sha(path)
    plan=dict(source=relative(source),workers=workers,environment=environment(),models=models,identity_control=control,
        warm_start=warm['request']['spec'],warm_record=relative(source/f"selection/candidate_{records.index(warm):03d}.json"),
        opponents=old['opponents'],groups=old['groups'],band=old['selection_band'],n=old['selection_n'],
        references=[old['anchor'],old['parent']],parent_donors=[old['parent'],alternative['request']['spec']],
        source_sha256=sources,input_sha256=inputs,
        scope='Consumed selection conditions now used for development. Preserve warm policy first .5 seconds and interceptor; only post-probe parent differs. Two explicit recombinations, no gradient training or final/unseen performance claim. Legacy policies and assessment failures unchanged.')
    immutable_json(out/'plan.json',plan)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    return plan


def run(out, source, workers):
    if (out/'runtime.json').exists() and process_live(read(out/'runtime.json')['pid']): raise ValueError('Probe already live')
    plan=freeze(out,source,workers)
    if (out/'completion.json').exists(): return read(out/'completion.json')
    before=resources()
    if before['commit_headroom_gib']<3+.42*workers+.3: raise ValueError('Insufficient commit headroom')
    write(out/'runtime.json',dict(pid=os.getpid(),workers=workers,started_at=datetime.now(timezone.utc).isoformat(),resources=before))
    warm=read(ROOT/plan['warm_record'])
    if not (out/'compatibility.json').exists():
        checks=[]
        # Both gate routes: reactive parent and candidate self-play interceptor.
        for identity in ('refine_2200',plan['warm_start']['id']):
            foe=next(f for f in plan['opponents'] if f['id']==identity)
            new=duel(plan['identity_control'],foe,plan['band'],4)
            old=next(r for r in warm['results'] if r['foe']==identity)
            expected=[e for e in old['episodes'] if plan['band']<=e['seed']<plan['band']+2]
            if new['episodes']!=expected: raise ValueError('Identity composition changed actual flight episodes')
            checks.append(dict(foe=identity,episodes=new['episodes']))
        immutable_json(out/'compatibility.json',dict(status='passed',games=8,checks=checks))
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker) as pool:
        records=Evaluation()(pool,plan['models'],plan['opponents'],plan['band'],plan['n'],out/'development')
    references=[read(source/f'selection/candidate_{i:03d}.json') for i in range(2)]
    result=dict(status='development_probe_complete',new_games=sum(r['metrics']['games'] for r in records),
        reused_reference_games=sum(r['metrics']['games'] for r in references),identity_compatibility_games=8,
        candidates=[dict(spec=r['request']['spec'],profile=profile(r['results'],plan['groups']),
            comparisons=[compare(r['results'],b['results'],plan['groups']) for b in references],
            change_vs_warm=compare(r['results'],warm['results'],plan['groups'])) for r in records],scope=plan['scope'])
    verify(plan);write(out/'completion.json',result)
    return result


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('runs/league_tournament_assess_20261007'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    try:run(a.out.resolve(),a.source.resolve(),a.workers)
    except Exception:write(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
