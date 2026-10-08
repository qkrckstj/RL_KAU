"""Separate verified NumPy pilots from neural/unknown submissions.

The original match runner and frozen experiments are untouched. Callers freeze
the approved source signatures and use pre-import thread limits. Unknown policy
code always runs in an ordinary worker. This pool accepts league duel and traced
duel jobs only; it is not a general executor.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import sys
import time

from tools.league_matches import Actor, duel, initialize_worker, summary
from tools.league_numpy_matches import (
    NumpyActor, MatchEnv as NumpyEnv, numpy_capable, duel as numpy_duel,
)
from tools.league_thread_benchmark import THREAD_KEYS, own_memory


def initialize(barrier, queue, specs, signatures, pure):
    if any(os.environ.get(key) != '1' for key in THREAD_KEYS):
        raise ValueError('Set thread limits before starting Python')
    initialize_worker()
    for spec in specs:
        NumpyActor(spec, signatures) if pure else Actor(spec)
    if pure and 'torch' in sys.modules:
        raise RuntimeError('Pure worker imported torch')
    queue.put(dict(**own_memory(), pure=pure, torch_imported='torch' in sys.modules))
    barrier.wait(timeout=180)


def ready():
    return os.getpid()


def execute(own, foe, band, n, signatures, traced):
    if not traced:
        result = numpy_duel(own, foe, band, n, signatures)
    else:
        from tools.league_damage_trace import TracedEnv
        from tools.grade import play

        class PureTracedEnv(TracedEnv):
            def __init__(self, own, foe, seat):
                NumpyEnv.__init__(self, own, foe, seat, signatures)

        if n <= 0 or n % 2:
            raise ValueError('Complete seat pairs required')
        started = time.perf_counter()
        episodes, traces = [], []
        for seat in ('red', 'blue'):
            env = PureTracedEnv(own, foe, seat)
            try:
                for seed in range(band, band+n//2):
                    row = play(env, env.own_action, seed, 1, seat)[0]
                    row['seat'] = seat
                    episodes.append(row)
                    traces.append(dict(seed=seed, seat=seat, **env.trace.finish()))
            finally:
                env.close()
        result = dict(own=own['id'], foe=foe['id'], band=band,
            episodes=episodes, summary=summary(episodes), damage_traces=traces,
            elapsed_seconds=time.perf_counter()-started)
    if 'torch' in sys.modules:
        raise RuntimeError('Pure match imported torch')
    return result


class SplitPool:
    def __init__(self, workers, neural_workers, signatures, pure_warmup, neural_warmup):
        if not 1 <= neural_workers < workers <= (os.cpu_count() or 1):
            raise ValueError('Invalid split worker counts')
        if any(os.environ.get(key) != '1' for key in THREAD_KEYS):
            raise ValueError('Set all thread limits before starting Python')
        self.signatures = tuple(signatures)
        self.workers, self.neural_workers = workers, neural_workers
        self.pure_warmup, self.neural_warmup = pure_warmup, neural_warmup
        self.pools, self.queues, self.initialized_workers = {}, [], []

    def __enter__(self):
        context = mp.get_context('spawn')
        pending = []
        try:
            for pure, count, specs in (
                (True, self.workers-self.neural_workers, self.pure_warmup),
                (False, self.neural_workers, self.neural_warmup),
            ):
                queue, barrier = context.Queue(), context.Barrier(count)
                self.queues.append(queue)
                pool = ProcessPoolExecutor(max_workers=count, mp_context=context,
                    initializer=initialize, initargs=(barrier, queue, specs, self.signatures, pure))
                self.pools[pure] = pool
                pending.extend(pool.submit(ready) for _ in range(count))
                self.initialized_workers.extend(queue.get(timeout=180) for _ in range(count))
            for future in pending:
                future.result()
            if len({row['pid'] for row in self.initialized_workers}) != self.workers:
                raise RuntimeError('Worker initialization incomplete')
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def submit(self, function, *args):
        if len(args) != 4:
            raise ValueError('Expected own, foe, band, n')
        if function is duel:
            traced = False
        else:
            from tools.league_repair_train import timed_observe
            if function is not timed_observe:
                raise ValueError('Unsupported match function')
            traced = True
        own, foe, band, n = args
        pure = all(numpy_capable(spec, self.signatures) for spec in (own, foe))
        if pure:
            return self.pools[True].submit(execute, own, foe, band, n, self.signatures, traced)
        return self.pools[False].submit(function, *args)

    def __exit__(self, *args):
        for pool in self.pools.values():
            pool.shutdown(wait=True, cancel_futures=bool(args and args[0]))
        for queue in self.queues:
            queue.close()
            queue.join_thread()
