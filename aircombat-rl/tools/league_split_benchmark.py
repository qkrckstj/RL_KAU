"""Matched full-flight benchmark of ordinary and split policy worker pools.

No policy selection or training. All conditions here are reused development
conditions. The same jobs run twice in reversed scenario order.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import ctypes
from ctypes import wintypes
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
import time
import traceback

from tools import league_thread_benchmark as base

SCENARIOS = (('ordinary8', 8, 0), ('split8', 8, 2),
             ('split12', 12, 2), ('split16', 16, 2))
# Explicitly inspected exported policies: only json/math/numpy/pathlib/zipfile.
APPROVED_NEW = (
    'fe8e02c8b1501225b5be352e10f8e29098095de6962e253e8706272bf88bb58d',
)


def cpu_times():
    values = [wintypes.FILETIME() for _ in range(3)]
    if not ctypes.windll.kernel32.GetSystemTimes(*(ctypes.byref(v) for v in values)):
        raise ctypes.WinError()
    idle, kernel, user = [(v.dwHighDateTime << 32)+v.dwLowDateTime for v in values]
    return idle, kernel+user


def frozen_plan(source, out):
    from tools.league_numpy_matches import code_signature, numpy_capable
    old = base.read(source/'plan.json')
    base.verify(old)
    if not (source/'analysis_auto/completion.json').exists():
        raise ValueError('Prior validation analysis is not complete')
    if base.read(source/'analysis_auto/completion.json')['status'] != 'analysis_complete':
        raise ValueError('Prior analysis did not succeed')
    previous = base.ROOT/'runs/league_repair_20261007/numpy_loading_single_thread_probe.json'
    approved = set(base.read(previous)['approved_signatures']) | set(APPROVED_NEW)
    temporal = old['temporal']['opponents']
    # Temporal source is the explicitly inspected local NumPy controller.
    expected_temporal = '6934d1e2e16b418e07ec5572f4bba60b72da5f8bfbe0388defdb33edfa1d323e'
    for spec in temporal:
        if base.sha(base.ROOT/spec['design']/'policy.py') != expected_temporal:
            raise ValueError('Unexpected temporal policy source')
        approved.add(code_signature(spec))
    own = old['chosen']
    opponents = old['opponents'] + temporal + [own]
    pure = [s for s in [own]+opponents if numpy_capable(s, approved)]
    neural = [s for s in opponents if not numpy_capable(s, approved)]
    if sorted(s['id'] for s in neural) != ['ddqn_s0', 'ddqn_s1']:
        raise ValueError('Unexpected neural routing roster')
    warm, seen = [], set()
    for spec in pure:
        key = code_signature(spec) if spec['kind'] == 'submission' else spec['kind']
        if key not in seen:
            warm.append(spec)
            seen.add(key)
    # All 54 foes, including both neural checkpoints and both seats. Eight trace
    # jobs exercise full damage timelines on diverse policies in the same trial.
    trace_ids = {'ace', 'refine_2200', 'ddqn_s0', 'ddqn_s1',
                 'temporal_extend_right', 'temporal_weave_left', own['id'], old['warm']['id']}
    jobs = [dict(own=own, foe=foe, band=170000000, n=4,
                 traced=foe['id'] in trace_ids) for foe in opponents]
    sources = dict(old['source_sha256'])
    inputs = dict(old['input_sha256'])
    sources.update(old['temporal']['source_sha256'])
    inputs.update(old['temporal']['input_sha256'])
    for name in ('tools/league_split_pool.py', 'tools/league_split_benchmark.py',
                 'tools/league_numpy_matches.py', 'tools/league_thread_benchmark.py'):
        sources[name] = base.sha(base.ROOT/name)
    for path in (source/'plan.json', source/'completion.json',
                 source/'analysis_auto/completion.json', previous):
        inputs[base.relative(path)] = base.sha(path)
    for spec in [own]+opponents:
        if spec['kind'] == 'submission':
            for path in list((base.ROOT/spec['design']).glob('*.py'))+[base.ROOT/spec['weights']]:
                inputs[base.relative(path)] = base.sha(path)
    plan = dict(source=base.relative(source), jobs=jobs, scenarios=SCENARIOS,
        approved_signatures=sorted(approved), pure_warmup=warm,
        neural_warmup=[own]+neural, environment=base.environment(),
        source_sha256=sources, input_sha256=inputs,
        games_per_trial=sum(j['n'] for j in jobs),
        rule='Exact official episodes and full damage traces; reversed repeats; >=3 GiB commit and >=2 GiB physical headroom. Fewest total workers within 3% of fastest median eligible match time.',
        scope='Execution benchmark only; reused conditions, not new policy evidence. Thread limits all 1 before Python starts. No GPU or physics change.')
    base.verify(plan)
    base.write(out/'plan.json', plan)
    for name in sources:
        target=out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base.ROOT/name, target)
    return plan


def comparable(results):
    return [{k:v for k,v in r.items() if k != 'elapsed_seconds'} for r in results]


def trial(plan_path, out, scenario):
    plan=base.read(plan_path)
    base.verify(plan)
    if base.environment() != plan['environment']:
        raise ValueError('Changed benchmark environment')
    if any(os.environ.get(k) != '1' for k in base.THREAD_KEYS):
        raise ValueError('Set thread limits before Python starts')
    name, workers, neural = next(s for s in SCENARIOS if s[0] == scenario)
    base.write(out/'runtime.json', dict(pid=os.getpid(), scenario=name))
    from tools.league_split_pool import SplitPool, initialize, ready
    from tools.league_matches import duel
    from tools.league_repair_train import timed_observe
    context=mp.get_context('spawn')
    queue=None
    start=time.perf_counter()
    with base.Sampler() as sampler:
        if neural:
            pool=SplitPool(workers, neural, plan['approved_signatures'],
                plan['pure_warmup'], plan['neural_warmup'])
        else:
            queue=context.Queue()
            pool=ProcessPoolExecutor(max_workers=workers, mp_context=context,
                initializer=initialize, initargs=(context.Barrier(workers), queue,
                    plan['neural_warmup']+plan['pure_warmup'], plan['approved_signatures'], False))
        try:
            with pool:
                if neural:
                    memories=pool.initialized_workers
                else:
                    warm=[pool.submit(ready) for _ in range(workers)]
                    memories=[queue.get(timeout=180) for _ in range(workers)]
                    for f in warm: f.result()
                base.write(out/'workers.json', memories)
                warmed=time.perf_counter()
                cpu_a=cpu_times()
                futures=[pool.submit(timed_observe if j['traced'] else duel,
                    j['own'], j['foe'], j['band'], j['n']) for j in plan['jobs']]
                results=[f.result() for f in futures]
                seconds=time.perf_counter()-warmed
                cpu_b=cpu_times()
        finally:
            if queue is not None:
                queue.close(); queue.join_thread()
    record=dict(scenario=name, workers=workers, neural_workers=neural,
        request=dict(plan_sha256=base.sha(plan_path), scenario=name),
        startup_seconds=warmed-start, match_seconds=seconds,
        total_seconds=time.perf_counter()-start, initialized_workers=memories,
        cpu_mean_percent=100*(1-(cpu_b[0]-cpu_a[0])/(cpu_b[1]-cpu_a[1])),
        resources=sampler.result, results=results,
        games=sum(len(r['episodes']) for r in results))
    base.verify(plan)
    base.write(out/'result.json', record)


def run(source, out):
    if out.exists():
        raise ValueError('Use a fresh benchmark output')
    plan=frozen_plan(source, out)
    base.write(out/'runtime.json', dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat()))
    records, skipped=[], []
    for repeat, order in enumerate((SCENARIOS, tuple(reversed(SCENARIOS)))):
        for name, workers, neural in order:
            # Conservative preflight; actual sampled resources determine eligibility.
            estimate=(.45*workers if not neural else .45*neural+.15*(workers-neural))+.2
            resource=base.resources()
            if resource['commit_headroom_gib'] < 2+estimate or resource['available_memory_gib'] < 2+estimate:
                skipped.append(dict(repeat=repeat, scenario=name, resources=resource,
                    estimated_increment_gib=estimate, reason='preflight_headroom'))
                base.write(out/f'r{repeat}_{name}/skipped.json', skipped[-1])
                continue
            folder=out/f'r{repeat}_{name}'
            folder.mkdir(parents=True)
            base.write(out/'progress.json', dict(repeat=repeat, scenario=name))
            with (folder/'stdout.log').open('w',encoding='utf-8') as stdout, (folder/'stderr.log').open('w',encoding='utf-8') as stderr:
                subprocess.run([sys.executable,'-X','utf8','-u','-m','tools.league_split_benchmark',
                    '--trial','--plan',str(out/'plan.json'),'--out',str(folder),'--scenario',name],
                    cwd=base.ROOT, env=base.trial_environment(os.environ,'single'),
                    stdout=stdout,stderr=stderr,creationflags=0x08000000,check=True)
            record=base.read(folder/'result.json')
            if records and comparable(record['results']) != comparable(records[0]['results']):
                raise ValueError('Loader/concurrency changed episode records or damage traces')
            records.append(dict(**record, repeat=repeat))
    grouped={name:[r for r in records if r['scenario']==name] for name,_,_ in SCENARIOS}
    if len(grouped['ordinary8']) != 2:
        raise ValueError('Matched original baseline missing')
    seconds={name:median(r['match_seconds'] for r in rows) for name,rows in grouped.items() if len(rows)==2}
    eligible={name:seconds[name] for name,rows in grouped.items() if name in seconds and all(
        r['resources']['minimum_commit_headroom_gib']>=3 and
        r['resources']['minimum_available_memory_gib']>=2 for r in rows)}
    if not eligible:
        raise ValueError('No configuration retains required memory headroom')
    selected=min((s for s in SCENARIOS if s[0] in eligible and eligible[s[0]]<=1.03*min(eligible.values())),
                 key=lambda s:(s[1],s[2]))
    result=dict(status='benchmark_complete', exact_episode_and_trace_equality=True,
        recommended=dict(scenario=selected[0], workers=selected[1], neural_workers=selected[2]),
        median_match_seconds=seconds,
        speedup_vs_ordinary8=seconds['ordinary8']/seconds[selected[0]],
        minimum_commit_headroom_gib={name:min(r['resources']['minimum_commit_headroom_gib'] for r in rows)
            for name,rows in grouped.items() if rows},
        cpu_mean_percent={name:median(r['cpu_mean_percent'] for r in rows) for name,rows in grouped.items() if rows},
        initialized_worker_private_gib={name:{kind:median(w['private_gib'] for r in rows
            for w in r['initialized_workers'] if w['pure']==pure)
            for kind,pure in (('pure',True),('normal',False))
            if any(w['pure']==pure for r in rows for w in r['initialized_workers'])}
            for name,rows in grouped.items() if rows},
        games_executed=sum(r['games'] for r in records), games_per_trial=plan['games_per_trial'],
        skipped=skipped, rule=plan['rule'], scope=plan['scope'])
    base.verify(plan)
    base.write(out/'completion.json', result)
    base.write(out/'progress.json', dict(stage='complete'))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('runs/league_separated_validate_20261007'))
    parser.add_argument('--trial', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--scenario', choices=[s[0] for s in SCENARIOS])
    args=parser.parse_args()
    try:
        trial(args.plan.resolve(), args.out.resolve(), args.scenario) if args.trial else run(args.source.resolve(), args.out.resolve())
    except BaseException:
        base.write(args.out/'failure.json', dict(traceback=traceback.format_exc()))
        raise
