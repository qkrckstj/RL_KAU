"""Measure CPU concurrency on identical official matches, then start a successor.

No policy selection from these matches. Wait for existing diagnostics to finish
so other project simulations do not confound the comparison. No new packages.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
import ctypes
from ctypes import wintypes
import multiprocessing as mp
import os
import threading
import time
import traceback

from tools.autolab_cem import ROOT, read, write, sha
from tools.league_matches import Actor, duel, initialize_worker
from tools.league_maneuver_train import environment, run as train
from tools.league_train import verify
from tools.league_validate import process_live


class Memory(ctypes.Structure):
    _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total',
        'page_available', 'virtual_total', 'virtual_available', 'extended')]


def resources():
    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        raise ctypes.WinError()
    value = lambda x: (x.dwHighDateTime << 32) + x.dwLowDateTime
    memory = Memory(); memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError()
    return value(idle), value(kernel) + value(user), memory.available / 2**30


class Sampler:
    def __enter__(self):
        self.stop = threading.Event(); self.rows = []; self.initial = resources()
        def sample():
            while not self.stop.wait(.5):
                self.rows.append(resources())
        self.thread = threading.Thread(target=sample, daemon=True); self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set(); self.thread.join(); self.rows.append(resources())
        a, b = self.initial, self.rows[-1]
        self.result = dict(cpu_mean_pct=100 * (1-(b[0]-a[0])/(b[1]-a[1])),
                          minimum_free_memory_gib=min(x[2] for x in self.rows))


def initialize(barrier, own, neural):
    initialize_worker()
    Actor(own); Actor(neural)  # Include actual policy/Torch imports in startup.
    barrier.wait(timeout=180)


def ready():
    return os.getpid()


def comparable(results):
    return [dict(own=x['own'], foe=x['foe'], band=x['band'], episodes=x['episodes']) for x in results]


def choose(records):
    grouped = {}
    for row in records:
        grouped.setdefault(row['workers'], []).append(row)
    eligible = {w: median(r['seconds'] for r in rows) for w, rows in grouped.items()
                if len(rows) == 2 and all(r['minimum_free_memory_gib'] >= 2 for r in rows)}
    if not eligible:
        raise ValueError('No configuration retained sufficient free memory')
    fastest = min(eligible.values())
    selected = min(w for w, seconds in eligible.items() if seconds <= fastest * 1.03)
    return dict(workers=selected, median_seconds=eligible,
                speedup_vs_three=eligible[3]/eligible[selected],
                rule='Fewest workers within 3% of fastest median, >=2 GiB free in both trials; two reversed-order rounds')


def run(args):
    out = args.out.resolve()
    if (out/'runtime.json').exists() and process_live(read(out/'runtime.json')['pid']):
        raise ValueError('Benchmark controller already live')
    if (out/'completion.json').exists():
        raise ValueError('Use a fresh benchmark output')
    old = read(args.previous/'plan.json'); verify(old)
    sources = dict(old['source_sha256'])
    for name in ('tools/league_worker_benchmark.py', 'tools/league_maneuver_train.py',
                 'tools/league_damage_trace.py', 'tools/league_damage_analysis.py', 'tools/league_cached_evaluation.py'):
        sources[name] = sha(ROOT/name)
    frozen_environment = environment()
    write(out/'runtime.json', dict(pid=os.getpid(), source_sha256=sources,
          environment=frozen_environment, successor=str(args.successor), simulation_workers_while_waiting=0))
    while not (args.damage/'analysis/completion.json').exists():
        if not process_live(args.after_pid):
            raise ValueError('Damage analysis exited without completion')
        write(out/'progress.json', dict(stage='waiting_for_diagnostics', pid=args.after_pid))
        time.sleep(5)
    if read(args.damage/'analysis/completion.json')['status'] != 'analysis_complete':
        raise ValueError('Damage analysis did not succeed')
    if environment() != frozen_environment or any(sha(ROOT/name) != v for name, v in sources.items()):
        raise ValueError('Queued benchmark source or environment changed')
    own = read(args.previous/'frozen_selection.json')['chosen']
    names = ('ace', 'evader', 'circler', 'refine_2201', 'ddqn_s1', 'round_05')
    foes = [next(x for x in old['opponents'] if x['id'] == name) for name in names]
    jobs = [(own, foe, 170000000 + block*100, 4) for block in range(4) for foe in foes]
    counts = [w for w in (3, 6, 9, 12, 16) if w <= (os.cpu_count() or 1)]
    write(out/'plan.json', dict(workers=counts, rounds=2, jobs=jobs, games_per_trial=96,
          source_sha256=sources, scope='Throughput only; same 96 matches repeated, not independent performance evidence'))
    baseline = None; records = []
    for repeat, order in enumerate((counts, list(reversed(counts)))):
        for workers in order:
            write(out/'progress.json', dict(stage='benchmark', repeat=repeat, workers=workers))
            context = mp.get_context('spawn'); barrier = context.Barrier(workers)
            started = time.perf_counter()
            with Sampler() as sampler:
                with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=initialize,
                                         initargs=(barrier, own, foes[4])) as pool:
                    warm = [pool.submit(ready) for _ in range(workers)]
                    for f in warm: f.result()
                    startup = time.perf_counter() - started
                    t = time.perf_counter()
                    pending = [pool.submit(duel, *job) for job in jobs]
                    results = [f.result() for f in pending]
                    seconds = time.perf_counter() - t
            batch = dict(repeat=repeat, workers=workers, seconds=seconds, startup_seconds=startup,
                         total_seconds=time.perf_counter()-started, games=96,
                         games_per_second=96/seconds, **sampler.result)
            write(out/f'r{repeat}_w{workers}.json', dict(timing=batch, results=results))
            if baseline is None: baseline = comparable(results)
            if comparable(results) != baseline:
                raise ValueError('Worker count changed official episode records')
            records.append(batch); write(out/'timings.json', records)
    decision = choose(records)
    write(out/'decision.json', dict(**decision, exact_episode_equality=True, records=records))
    if environment() != frozen_environment or any(sha(ROOT/name) != v for name, v in sources.items()):
        raise ValueError('Benchmark source or environment changed')
    write(out/'completion.json', dict(status='benchmark_complete', **decision))
    write(out/'progress.json', dict(stage='successor_training', workers=decision['workers']))
    train(args.successor, args.previous, args.damage, workers=decision['workers'])
    write(out/'progress.json', dict(stage='successor_complete'))


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--successor', type=Path, required=True)
    p.add_argument('--previous', type=Path, default=Path('runs/league_interception_train_20261006'))
    p.add_argument('--damage', type=Path, default=Path('runs/league_damage_trace_20261006'))
    p.add_argument('--after-pid', type=int, required=True)
    args = p.parse_args()
    try:
        run(args)
    except Exception:
        write(args.out/'failure.json', dict(traceback=traceback.format_exc()))
        raise
