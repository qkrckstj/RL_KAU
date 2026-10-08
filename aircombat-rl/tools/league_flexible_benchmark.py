"""Queue a bounded static/flexible comparison after a verified native process exit.

Only the standard library is imported while waiting. Reuses the prior 216
benchmark games twice per backend; no new policy selection or held-out data.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
import ctypes
from ctypes import wintypes
import os
import shutil
import subprocess
import sys
import time
import traceback

from tools import league_thread_benchmark as base
from tools.league_split_benchmark import cpu_times, comparable


def recommendation(records):
    grouped = {name: [r for r in records if r['scenario'] == name]
               for name in ('static16', 'flexible16')}
    if any(len(rows) != 2 for rows in grouped.values()):
        raise ValueError('Two matched trials per backend required')
    elapsed = {name: median(r['match_seconds'] for r in rows) for name, rows in grouped.items()}
    eligible = all(r['resources']['minimum_commit_headroom_gib'] >= 3 and
                   r['resources']['minimum_available_memory_gib'] >= 2 for r in records)
    per_repeat_faster = all(next(r['match_seconds'] for r in grouped['flexible16'] if r['repeat'] == i) <
                            next(r['match_seconds'] for r in grouped['static16'] if r['repeat'] == i)
                            for i in (0, 1))
    adopt = eligible and per_repeat_faster and elapsed['flexible16'] <= .97*elapsed['static16']
    return dict(recommended='flexible16' if adopt else 'static16',
                median_match_seconds=elapsed, speedup=elapsed['static16']/elapsed['flexible16'],
                memory_eligible=eligible, faster_in_both_repeats=per_repeat_faster,
                practical_threshold_met=elapsed['flexible16'] <= .97*elapsed['static16'])


def prepare(source, predecessor, out, pid):
    old = base.read(source/'plan.json'); base.verify(old)
    previous = base.read(predecessor/'plan.json'); base.verify(previous)
    runtime = base.read(predecessor/'runtime.json')
    if runtime['pid'] != pid or any(os.environ.get(k) != '1' for k in base.THREAD_KEYS):
        raise ValueError('Wrong predecessor or missing pre-import thread limits')
    completed = base.read(source/'completion.json')
    if not completed['exact_episode_and_trace_equality']:
        raise ValueError('Previous execution benchmark is not qualified')
    if old['environment'] != base.environment():
        raise ValueError('Prior benchmark environment changed')
    sources = dict(old['source_sha256']); inputs = dict(old['input_sha256'])
    for target, extra in ((sources, previous['source_sha256']), (inputs, previous['input_sha256'])):
        for name, digest in extra.items():
            if name in target and target[name] != digest:
                raise ValueError('Conflicting frozen dependency')
            target[name] = digest
    for name in ('tools/league_flexible_pool.py', 'tools/league_flexible_benchmark.py',
                 'tests/test_league_flexible_pool.py', 'tests/test_league_flexible_benchmark.py'):
        sources[name] = base.sha(base.ROOT/name)
    reference = source/'r0_split16/result.json'
    for path in (source/'plan.json', source/'completion.json', reference, predecessor/'plan.json'):
        inputs[base.relative(path)] = base.sha(path)
    plan = dict(jobs=old['jobs'], approved_signatures=old['approved_signatures'],
        pure_warmup=old['pure_warmup'], neural_warmup=old['neural_warmup'],
        reference_result=base.relative(reference), predecessor=base.relative(predecessor),
        after_pid=pid, source_started_at=runtime['started_at'], environment=base.environment(),
        workers=16, neural_workers=2, orders=[['static16', 'flexible16'], ['flexible16', 'static16']],
        games_per_trial=sum(j['n'] for j in old['jobs']), source_sha256=sources, input_sha256=inputs,
        rule='Exact records including selected full damage traces against previous reference and every trial; flexible must be faster in both repeats and at least3% faster by median, with >=3GiB commit and >=2GiB physical headroom.',
        scope='Execution comparison on reused conditions, 864 total games. No GPU, physics change, policy learning, new performance claim or GitHub publication. No active execution profile is changed automatically.')
    base.verify(plan); base.write(out/'plan.json', plan)
    for name in sources:
        path=out/'source_snapshot'/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base.ROOT/name, path)
    return plan


def wait_for_predecessor(plan, out):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000|0x1000, False, plan['after_pid'])
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        values = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(v) for v in values)):
            raise ctypes.WinError(ctypes.get_last_error())
        created = ((values[0].dwHighDateTime<<32)+values[0].dwLowDateTime)/1e7-11644473600
        if abs(created-datetime.fromisoformat(plan['source_started_at']).timestamp()) > 10:
            raise ValueError('Predecessor PID has a different creation time')
        base.write(out/'runtime.json', dict(pid=os.getpid(), after_pid=plan['after_pid'],
            started_at=datetime.now(timezone.utc).isoformat(), source_process_created_unix=created,
            simulation_workers_while_waiting=0, numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules, own_memory=base.own_memory()))
        print('Waiting on verified predecessor native handle; no simulation workers.', flush=True)
        while True:
            status = kernel.WaitForSingleObject(handle, 45000)
            if status == 0: break
            if status != 258: raise ctypes.WinError(ctypes.get_last_error())
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if code.value != 0: raise ValueError(f'Predecessor exited with code{code.value}')
    finally:
        kernel.CloseHandle(handle)
    predecessor = base.ROOT/plan['predecessor']
    if not (predecessor/'completion.json').exists() or (predecessor/'failure.json').exists():
        raise ValueError('Predecessor did not complete cleanly')
    if base.read(predecessor/'completion.json')['status'] not in (
            'repair_profile_passed', 'repair_profile_failed', 'no_development_improvement'):
        raise ValueError('Unexpected predecessor terminal state')
    base.verify(plan)
    if base.environment() != plan['environment']: raise ValueError('Environment changed while waiting')


def trial(plan_path, out, name):
    from tools.league_split_pool import SplitPool
    from tools.league_flexible_pool import FlexiblePool
    from tools.league_matches import duel
    from tools.league_repair_train import timed_observe
    plan = base.read(plan_path); base.verify(plan)
    if base.environment() != plan['environment']: raise ValueError('Changed trial environment')
    pool_type = {'static16': SplitPool, 'flexible16': FlexiblePool}[name]
    started = time.perf_counter()
    with base.Sampler() as sampler:
        with pool_type(16, 2, plan['approved_signatures'], plan['pure_warmup'], plan['neural_warmup']) as pool:
            workers = pool.initialized_workers
            base.write(out/'workers.json', workers)
            warmed = time.perf_counter(); cpu_a = cpu_times()
            futures = [pool.submit(timed_observe if j['traced'] else duel,
                j['own'], j['foe'], j['band'], j['n']) for j in plan['jobs']]
            results = [future.result() for future in futures]
            seconds = time.perf_counter()-warmed; cpu_b = cpu_times()
            routing = pool.router.stats() if name == 'flexible16' else None
    if comparable(results) != comparable(base.read(base.ROOT/plan['reference_result'])['results']):
        raise ValueError('Trial differs from previously verified official records')
    base.verify(plan)
    base.write(out/'result.json', dict(scenario=name, initialized_workers=workers,
        request=dict(plan_sha256=base.sha(plan_path), scenario=name), startup_seconds=warmed-started,
        match_seconds=seconds, total_seconds=time.perf_counter()-started,
        cpu_mean_percent=100*(1-(cpu_b[0]-cpu_a[0])/(cpu_b[1]-cpu_a[1])),
        resources=sampler.result, results=results, routing=routing,
        games=sum(len(r['episodes']) for r in results)))


def run(source, predecessor, out, pid):
    plan = prepare(source, predecessor, out, pid)
    wait_for_predecessor(plan, out)
    records = []
    for repeat, order in enumerate(plan['orders']):
        for name in order:
            resource = base.resources()
            if min(resource['commit_headroom_gib'], resource['available_memory_gib']) < 5.2:
                raise ValueError(f'Insufficient preflight memory headroom: {resource}')
            folder=out/f'r{repeat}_{name}'; folder.mkdir()
            base.write(out/'progress.json', dict(stage='benchmark', repeat=repeat, scenario=name))
            with (folder/'stdout.log').open('w', encoding='utf-8') as stdout, (folder/'stderr.log').open('w', encoding='utf-8') as stderr:
                subprocess.run([sys.executable, '-X', 'utf8', '-u', '-m', 'tools.league_flexible_benchmark',
                    '--trial', '--plan', str(out/'plan.json'), '--out', str(folder), '--scenario', name],
                    cwd=base.ROOT, env=base.trial_environment(os.environ, 'single'),
                    stdout=stdout, stderr=stderr, creationflags=0x08000000, check=True)
            record = base.read(folder/'result.json')
            if records and comparable(record['results']) != comparable(records[0]['results']):
                raise ValueError('Backend changed records across repeats')
            records.append(dict(**record, repeat=repeat))
    result = dict(status='benchmark_complete', **recommendation(records),
        exact_episode_and_trace_equality=True, games_executed=sum(r['games'] for r in records),
        routing=[dict(repeat=r['repeat'], **r['routing']) for r in records if r['routing'] is not None],
        rule=plan['rule'], scope=plan['scope'])
    base.verify(plan); base.write(out/'completion.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--after-run', type=Path)
    parser.add_argument('--after-pid', type=int)
    parser.add_argument('--source', type=Path, default=Path('runs/league_split_benchmark_20261007'))
    parser.add_argument('--trial', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--scenario', choices=('static16', 'flexible16'))
    args = parser.parse_args()
    if not args.trial and (args.out.exists() or args.after_run is None or args.after_pid is None):
        parser.error('Provide a fresh output and explicit predecessor run/PID')
    try:
        if args.trial: trial(args.plan.resolve(), args.out.resolve(), args.scenario)
        else: run(args.source.resolve(), args.after_run.resolve(), args.out.resolve(), args.after_pid)
    except BaseException:
        base.write(args.out/'failure.json', dict(traceback=traceback.format_exc()))
        raise
