"""Measure 2/4/6 ordinary workers within16 on the observed mixed screen roster.

No learning or policy selection. Requires completed assessment and repeats the
same consumed games in reversed configuration order with exact record checks.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
import os
import shutil
import subprocess
import sys
import traceback
from tools import league_thread_benchmark as io
from tools import league_split_benchmark as benchmark
from tools.league_numpy_matches import numpy_capable

SCENARIOS = (('ordinary2', 16, 2), ('ordinary4', 16, 4), ('ordinary6', 16, 6))


def freeze(source, out):
    assert not out.exists()
    assert all(os.environ.get(k) == '1' for k in io.THREAD_KEYS)
    assert not io.process_live(io.read(source/'runtime.json')['pid'])
    old = io.read(source/'plan.json'); io.verify(old)
    raw = io.read(source/'raw_budget_screen.json')
    assert raw['status'] == 'budget_screen_raw_verified'
    for p, h in raw['input_sha256'].items():
        assert io.sha(io.ROOT/p) == h
    own = old['roles']['candidate'][0]
    foes = old['opponents']; assert len(foes) == 56
    ordinary = [s for s in foes if not numpy_capable(s, old['approved_signatures'])]
    assert len(ordinary) == 14 and numpy_capable(own, old['approved_signatures'])
    trace_ids = {s['id'] for s in ordinary[:4]}
    trace_ids.update(s['id'] for s in foes if s['id'] in ('ace', 'evader', 'temporal_extend_left'))
    trace_ids.add(foes[-1]['id'])
    jobs = [dict(own=own, foe=s, band=old['band'], n=4, traced=s['id'] in trace_ids) for s in foes]
    sources = dict(old['source_sha256'])
    for p in [Path(__file__).resolve(), Path(benchmark.__file__).resolve()]:
        sources[io.relative(p)] = io.sha(p)
    inputs = dict(old['input_sha256'])
    for p in [source/'plan.json', source/'completion.json', source/'raw_budget_screen.json']:
        inputs[io.relative(p)] = io.sha(p)
    plan = dict(source=io.relative(source), jobs=jobs, scenarios=SCENARIOS,
        approved_signatures=old['approved_signatures'], pure_warmup=old['pure_warmup'],
        neural_warmup=old['neural_warmup'], source_sha256=sources, input_sha256=inputs,
        environment=io.environment(), games_per_trial=224, ordinary_jobs=14, pure_jobs=42,
        rule='Two reversed trials of16total workers with2,4,6ordinary workers. Exact episode/summary and selected full damage trace equality. Minimum3GiBcommit and2GiBphysical headroom. Fewest ordinary workers within3percent of fastest eligible median match time.',
        scope='Execution benchmark of one fixed sampled PPO against56known opponents on two consumed110M ICs/bothseats.1344executions if all configurations eligible. Not additional independent policy evidence,not a universal speedup estimate. No GPU/physics/policy changes or GitHub upload.')
    io.verify(plan); io.write(out/'plan.json', plan)
    for name in sources:
        dest=out/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(io.ROOT/name, dest)
    return plan


def run(source, out):
    plan = freeze(source, out)
    io.write(out/'runtime.json', dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat()))
    records, skipped = [], []
    for repeat, order in enumerate((SCENARIOS, tuple(reversed(SCENARIOS)))):
        for name, workers, neural in order:
            estimate=.45*neural+.15*(workers-neural)+.2
            resources=io.resources(); folder=out/f'r{repeat}_{name}'
            if min(resources[k] for k in ('commit_headroom_gib','available_memory_gib')) < 2+estimate:
                item=dict(repeat=repeat, scenario=name, resources=resources, reason='startup_headroom')
                skipped.append(item); io.write(folder/'skipped.json', item); continue
            folder.mkdir(parents=True)
            print('pool benchmark', repeat, name, flush=True)
            with (folder/'stdout.log').open('w',encoding='utf-8') as stdout, (folder/'stderr.log').open('w',encoding='utf-8') as stderr:
                subprocess.run([sys.executable,'-X','utf8','-u','-m','tools.league_budget_pool_benchmark',
                    '--trial','--plan',str(out/'plan.json'),'--out',str(folder),'--scenario',name],
                    cwd=io.ROOT,env=io.trial_environment(os.environ,'single'),stdout=stdout,stderr=stderr,
                    creationflags=0x08000000,check=True)
            row=io.read(folder/'result.json')
            assert row['games']==224 and row['request']==dict(plan_sha256=io.sha(out/'plan.json'),scenario=name)
            if records:
                assert benchmark.comparable(row['results'])==benchmark.comparable(records[0]['results'])
            records.append(dict(row,repeat=repeat))
    grouped={name:[r for r in records if r['scenario']==name] for name,_,_ in SCENARIOS}
    assert len(grouped['ordinary2'])==2, 'Baseline repetitions missing'
    times={k:median(r['match_seconds'] for r in rs) for k,rs in grouped.items() if len(rs)==2}
    eligible={k:v for k,v in times.items() if all(r['resources']['minimum_commit_headroom_gib']>=3
        and r['resources']['minimum_available_memory_gib']>=2 for r in grouped[k])}
    assert eligible, 'No eligible worker allocation'
    selected=min((s for s in SCENARIOS if s[0] in eligible and eligible[s[0]]<=1.03*min(eligible.values())),key=lambda s:s[2])
    io.verify(plan)
    result=dict(status='mixed_pool_benchmark_verified',exact_episode_and_trace_equality=True,
        recommended=dict(scenario=selected[0],workers=selected[1],neural_workers=selected[2]),
        median_match_seconds=times,median_startup_seconds={k:median(r['startup_seconds'] for r in rs) for k,rs in grouped.items() if rs},
        speedup_vs_ordinary2=times['ordinary2']/times[selected[0]],
        minimum_commit_headroom_gib={k:min(r['resources']['minimum_commit_headroom_gib'] for r in rs) for k,rs in grouped.items() if rs},
        minimum_available_memory_gib={k:min(r['resources']['minimum_available_memory_gib'] for r in rs) for k,rs in grouped.items() if rs},
        games_executed=sum(r['games'] for r in records),skipped=skipped,rule=plan['rule'],scope=plan['scope'],
        input_sha256={io.relative(p):io.sha(p) for p in [out/'plan.json',*out.glob('r*/result.json')]})
    io.write(out/'completion.json',result)
    print('mixed pool benchmark complete', result['recommended'], result['speedup_vs_ordinary2'], flush=True)


if __name__ == '__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,default=Path('runs/league_rate_budget_assess_20261008'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--trial',action='store_true')
    p.add_argument('--plan',type=Path);p.add_argument('--scenario',choices=[s[0] for s in SCENARIOS]);a=p.parse_args()
    benchmark.SCENARIOS=SCENARIOS
    try:
        if a.trial: benchmark.trial(a.plan.resolve(),a.out.resolve(),a.scenario)
        else: run(a.source.resolve(),a.out.resolve())
    except BaseException:
        io.write(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
