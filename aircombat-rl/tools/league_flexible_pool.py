"""Bounded queues let ordinary workers help verified NumPy jobs when idle.

Experimental backend: no active/frozen run imports this module. Unknown and
neural policies never enter a pure worker. Ordinary workers prefer queued
neural jobs; a running helper job is not preempted. Measure actual throughput
and record equality before adopting this for any new training experiment.
"""
from collections import deque
from concurrent.futures import Future
from threading import Condition, RLock

from tools.league_matches import duel
from tools.league_numpy_matches import numpy_capable
from tools.league_split_pool import SplitPool, execute


class BoundedRouter:
    """Dispatch at most capacity jobs to each of two heterogeneous executors."""

    def __init__(self, capacities, dispatch):
        if set(capacities) != {True, False} or min(capacities.values()) < 1:
            raise ValueError('Both pools need positive capacities')
        self.capacities = dict(capacities)
        self.dispatch = dispatch
        self.pending = {True: deque(), False: deque()}
        self.inflight = {True: 0, False: 0}
        self.condition = Condition(RLock())
        self.accepting = True
        self.pumping = False
        self.counts = dict(pure_jobs=0, ordinary_jobs=0, helper_jobs=0,
                           completed=0, failed=0, cancelled=0)

    def submit(self, pure, payload):
        future = Future()
        with self.condition:
            if not self.accepting:
                raise RuntimeError('Router is shutting down')
            self.pending[bool(pure)].append((future, payload))
            self._pump()
        return future

    def _launch(self, pool_kind, job_kind):
        future, payload = self.pending[job_kind].popleft()
        if not future.set_running_or_notify_cancel():
            self.counts['cancelled'] += 1
            return
        self.inflight[pool_kind] += 1
        key = 'pure_jobs' if pool_kind else ('helper_jobs' if job_kind else 'ordinary_jobs')
        self.counts[key] += 1
        try:
            actual = self.dispatch(pool_kind, payload)
        except BaseException as error:
            self.inflight[pool_kind] -= 1
            self.counts['failed'] += 1
            future.set_exception(error)
            return

        def finished(actual):
            with self.condition:
                self.inflight[pool_kind] -= 1
                try:
                    value = actual.result()
                except BaseException as error:
                    self.counts['failed'] += 1
                    future.set_exception(error)
                else:
                    self.counts['completed'] += 1
                    future.set_result(value)
                self._pump()
                self.condition.notify_all()

        actual.add_done_callback(finished)

    def _pump(self):
        # A completed Future can invoke its callback synchronously. The outer
        # loop drains available work, preventing recursive dispatch chains.
        if self.pumping:
            return
        self.pumping = True
        try:
            while self.pending[True] and self.inflight[True] < self.capacities[True]:
                self._launch(True, True)
            while self.inflight[False] < self.capacities[False]:
                if self.pending[False]:
                    self._launch(False, False)
                elif self.pending[True]:
                    self._launch(False, True)
                else:
                    break
        finally:
            self.pumping = False

    def close(self, cancel=False):
        with self.condition:
            self.accepting = False
            if cancel:
                for queue in self.pending.values():
                    while queue:
                        future, _ = queue.popleft()
                        future.cancel()
                        future.set_running_or_notify_cancel()
                        self.counts['cancelled'] += 1
            self._pump()
            while any(self.pending.values()) or any(self.inflight.values()):
                self.condition.wait()

    def stats(self):
        with self.condition:
            return dict(**self.counts, queued_pure=len(self.pending[True]),
                        queued_ordinary=len(self.pending[False]),
                        active_pure=self.inflight[True], active_ordinary=self.inflight[False])


class FlexiblePool(SplitPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router = None

    def __enter__(self):
        if self.router is not None:
            raise RuntimeError('Pool cannot be entered twice')
        super().__enter__()
        self.router = BoundedRouter(
            {True: self.workers-self.neural_workers, False: self.neural_workers},
            self._dispatch)
        return self

    def _dispatch(self, pure, payload):
        function, args, traced = payload
        if pure:
            return self.pools[True].submit(execute, *args, self.signatures, traced)
        # Ordinary workers use the original loader, including for helper jobs.
        # The pure execute() intentionally rejects an imported torch module.
        return self.pools[False].submit(function, *args)

    def submit(self, function, *args):
        if self.router is None:
            raise RuntimeError('Enter the pool before submitting jobs')
        if len(args) != 4:
            raise ValueError('Expected own, foe, band, n')
        if function is duel:
            traced = False
        else:
            from tools.league_repair_train import timed_observe
            if function is not timed_observe:
                raise ValueError('Unsupported match function')
            traced = True
        own, foe, _, _ = args
        pure = all(numpy_capable(spec, self.signatures) for spec in (own, foe))
        return self.router.submit(pure, (function, args, traced))

    def __exit__(self, *args):
        try:
            if self.router is not None:
                self.router.close(cancel=bool(args and args[0]))
        finally:
            super().__exit__(*args)
