"""Smaller baseline and measured-resource worker sweep after v1 preflight failure.

The failed v1 plan/result remain unchanged. Start no competing simulations.
Reuse its identical-game trial implementation, with a new frozen scenario plan.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
from statistics import median
import subprocess
import sys
import traceback
from tools import league_thread_benchmark as base

SCENARIOS = (('default_w2', 'default', 2), ('single_w2', 'single', 2),
             ('single_w8', 'single', 8), ('single_w12', 'single', 12), ('single_w16', 'single', 16))


def decision(records, skipped):
    groups = {name: [r for r in records if r['scenario'] == name] for name, _, _ in SCENARIOS}
    if not records or any(base.comparable(r['results']) != base.comparable(records[0]['results']) for r in records):
        raise ValueError('Missing or changed episode records')
    if any(len(groups[k]) != 2 for k in ('default_w2', 'single_w2')):
        raise ValueError('Matched two-worker comparison incomplete')
    seconds = {k: median(r['match_seconds'] for r in rs) for k, rs in groups.items() if len(rs) == 2}
    private = {k: median(sum(w['private_gib'] for w in r['initialized_workers']) / len(r['initialized_workers'])
        for r in rs) for k, rs in groups.items() if len(rs) == 2}
    eligible = {n: seconds[k] for k, mode, n in SCENARIOS if mode == 'single' and k in seconds and all(
        r['resources']['minimum_commit_headroom_gib'] >= 3 and r['resources']['minimum_available_memory_gib'] >= 2
        for r in groups[k])}
    workers = min((n for n, t in eligible.items() if t <= 1.03 * min(eligible.values())), default=None)
    return dict(status='benchmark_complete', exact_episode_equality=True, recommended_workers=workers,
        median_match_seconds=seconds, median_initialized_worker_private_gib=private,
        matched_two_worker_speed_ratio=seconds['default_w2'] / seconds['single_w2'],
        skipped=skipped, preimport_thread_environment={k:'1' for k in base.THREAD_KEYS},
        minimum_observed_commit_headroom_gib={k:min(r['resources']['minimum_commit_headroom_gib'] for r in rs)
            for k,rs in groups.items() if rs},
        rule='Two reversed repeats with exact episodes; fewest single-thread workers within 3% of fastest eligible; >=3 GiB commit and >=2 GiB physical headroom in both repeats.',
        scope='New plan after v1 failed without matches. Smaller two-worker baseline. This is execution performance, not independent policy evidence. Compare thread setting at two workers; no matched default-16 speed claim.')


def freeze(out, failed):
    prior = base.read(failed / 'plan.json'); base.verify(prior)
    if base.process_live(base.read(failed / 'runtime.json')['pid']):
        raise ValueError('Prior benchmark still live')
    if not (failed / 'failure.json').exists() or list(failed.glob('r*/result.json')):
        raise ValueError('Expected v1 preflight failure without any trial')
    for name in ('search', 'audit'):
        folder = base.ROOT / prior[name]
        if base.process_live(base.read(folder / 'runtime.json')['pid']) or not (folder / 'completion.json').exists():
            raise ValueError('Predecessor not terminal')
    if (out / 'plan.json').exists():
        plan = base.read(out / 'plan.json'); base.verify(plan)
        if plan['failed_benchmark'] != base.relative(failed) or plan['environment'] != base.environment():
            raise ValueError('Changed benchmark environment')
        return plan
    plan = dict(prior)
    plan.update(failed_benchmark=base.relative(failed), scenarios=[list(s) for s in SCENARIOS],
        environment=base.environment(), reserve_rule='Pilot preflight estimates 1 GiB per default process, .35 GiB per limited process, plus 2 GiB commit; >=2 GiB physical free after estimate. Single-process private estimate then uses maximum prior measured per-worker private +.05 GiB. Final eligibility requires 3 GiB commit and 2 GiB physical in both repeats. Skip infeasible trials explicitly.',
        maximum_games=112*len(SCENARIOS)*2)
    plan['source_sha256'] = dict(prior['source_sha256'])
    plan['source_sha256']['tools/league_thread_benchmark_v2.py'] = base.sha(__file__)
    plan['input_sha256'] = dict(prior['input_sha256'])
    for name in ('plan.json', 'failure.json'):
        plan['input_sha256'][base.relative(failed / name)] = base.sha(failed / name)
    base.write(out / 'plan.json', plan)
    for name in plan['source_sha256']:
        target=out/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base.ROOT/name,target)
    return plan


def run(out, failed):
    if (out / 'runtime.json').exists() and base.process_live(base.read(out / 'runtime.json')['pid']):
        raise ValueError('Benchmark already live')
    plan=freeze(out,failed)
    if (out/'completion.json').exists(): return base.read(out/'completion.json')
    base.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat()))
    records=[];skipped=[]
    for repeat,order in enumerate((SCENARIOS,tuple(reversed(SCENARIOS)))):
        for name,mode,workers in order:
            folder=out/f'r{repeat}_{name}';path=folder/'result.json'
            if not path.exists():
                known=[w['private_gib'] for r in records if r['scenario'].startswith(mode+'_') for w in r['initialized_workers']]
                per_process=max(known)+.05 if known else (1. if mode=='default' else .35)
                estimate=per_process*(workers+1)
                resource=base.resources()
                if resource['commit_headroom_gib'] < 2+estimate or resource['available_memory_gib'] < 2+estimate:
                    row=dict(repeat=repeat,scenario=name,estimated_increment_gib=estimate,resources=resource,reason='preflight_headroom')
                    base.write(folder/'skipped.json',row);skipped.append(row);continue
                base.write(out/'progress.json',dict(stage='benchmark',repeat=repeat,scenario=name))
                folder.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,'-X','utf8','-u','-m','tools.league_thread_benchmark','--trial',
                    '--out',str(folder),'--plan',str(out/'plan.json'),'--mode',mode,'--workers',str(workers)]
                with (folder/'stdout.log').open('w',encoding='utf-8') as stdout,(folder/'stderr.log').open('w',encoding='utf-8') as stderr:
                    subprocess.run(command,cwd=base.ROOT,env=base.trial_environment(os.environ,mode),
                        stdout=stdout,stderr=stderr,creationflags=0x08000000,check=True)
            record=base.read(path)
            if record['request'] != dict(plan_sha256=base.sha(out/'plan.json'),mode=mode,workers=workers):
                raise ValueError('Changed trial request')
            records.append(dict(**record,scenario=name,repeat=repeat))
    result=decision(records,skipped);base.verify(plan)
    base.write(out/'completion.json',result);base.write(out/'progress.json',dict(stage='complete'))
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--failed',type=Path,default=Path('runs/league_thread_benchmark_20261007'))
    args=parser.parse_args()
    try:run(args.out.resolve(),args.failed.resolve())
    except Exception:
        base.write(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
