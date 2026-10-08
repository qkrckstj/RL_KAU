from concurrent.futures import Future, CancelledError
from threading import Event, Thread

import pytest

from tools.league_flexible_pool import BoundedRouter, FlexiblePool


class ManualDispatch:
    def __init__(self):
        self.jobs = []

    def __call__(self, pure, payload):
        future = Future()
        self.jobs.append((pure, payload, future))
        return future

    def finish(self, payload):
        next(f for _, p, f in self.jobs if p == payload).set_result(payload)


def test_idle_ordinary_help_and_neural_priority_without_pure_contamination():
    dispatch = ManualDispatch()
    router = BoundedRouter({True: 2, False: 1}, dispatch)
    futures = [router.submit(True, f'p{i}') for i in range(5)]
    neural = router.submit(False, 'neural')
    assert [(p, x) for p, x, _ in dispatch.jobs] == [(True, 'p0'), (True, 'p1'), (False, 'p2')]
    dispatch.finish('p2')
    assert dispatch.jobs[-1][:2] == (False, 'neural')
    dispatch.finish('p0')
    assert dispatch.jobs[-1][:2] == (True, 'p3')
    dispatch.finish('neural')
    assert dispatch.jobs[-1][:2] == (False, 'p4')
    for name in ('p1', 'p3', 'p4'):
        dispatch.finish(name)
    router.close()
    assert [f.result() for f in futures] == [f'p{i}' for i in range(5)]
    assert neural.result() == 'neural'
    assert router.stats() == dict(pure_jobs=3, ordinary_jobs=1, helper_jobs=2,
        completed=6, failed=0, cancelled=0, queued_pure=0, queued_ordinary=0,
        active_pure=0, active_ordinary=0)


def test_cancel_shutdown_drains_running_jobs_and_rejects_new_work():
    dispatch = ManualDispatch()
    router = BoundedRouter({True: 1, False: 1}, dispatch)
    first = router.submit(True, 'first')
    second = router.submit(True, 'second')
    cancelled = router.submit(True, 'cancelled')
    assert cancelled.cancel()
    neural = router.submit(False, 'queued_neural')
    closed = Event()
    thread = Thread(target=lambda: (router.close(cancel=True), closed.set()), daemon=True)
    thread.start()
    # Synchronize with close acquiring the router lock, not a scheduling sleep.
    with router.condition:
        if router.accepting:
            router.condition.wait_for(lambda: not router.accepting, timeout=2)
    assert not router.accepting
    assert not closed.is_set()
    with pytest.raises(RuntimeError):
        router.submit(True, 'late')
    dispatch.finish('first')
    dispatch.finish('second')
    assert closed.wait(2)
    thread.join(2)
    assert first.result() == 'first' and second.result() == 'second'
    for future in (cancelled, neural):
        with pytest.raises(CancelledError):
            future.result()
    assert len(dispatch.jobs) == 2


def test_dispatch_and_worker_failures_do_not_strand_queue():
    dispatch = ManualDispatch()
    router = BoundedRouter({True: 1, False: 1}, dispatch)
    first = router.submit(True, 'first')
    second = router.submit(True, 'second')
    third = router.submit(False, 'third')
    dispatch.jobs[1][2].set_exception(ValueError('worker failure'))
    assert dispatch.jobs[-1][:2] == (False, 'third')
    dispatch.finish('third')
    dispatch.finish('first')
    router.close()
    with pytest.raises(ValueError, match='worker failure'):
        second.result()
    assert first.result() == 'first' and third.result() == 'third'

    def raising(pool, payload):
        raise OSError('submit failure')
    router = BoundedRouter({True: 1, False: 1}, raising)
    failed = router.submit(False, 'x')
    router.close()
    with pytest.raises(OSError, match='submit failure'):
        failed.result()


def test_synchronous_completion_and_cancelled_queued_future():
    def immediate(pool, payload):
        future = Future(); future.set_result(payload)
        return future
    router = BoundedRouter({True: 1, False: 1}, immediate)
    assert [router.submit(i % 2 == 0, i).result() for i in range(2000)] == list(range(2000))
    router.close()

    dispatch = ManualDispatch()
    router = BoundedRouter({True: 1, False: 1}, dispatch)
    router.submit(True, 'first'); router.submit(True, 'second')
    cancelled = router.submit(False, 'cancelled')
    assert cancelled.cancel()
    queued = router.submit(True, 'queued')
    dispatch.finish('second')
    assert dispatch.jobs[-1][:2] == (False, 'queued')
    dispatch.finish('queued'); dispatch.finish('first'); router.close()
    assert queued.result() == 'queued'
    assert all(payload != 'cancelled' for _, payload, _ in dispatch.jobs)


def test_adapter_routes_helpers_through_original_function(monkeypatch):
    from tools import league_flexible_pool as module
    from tools.league_repair_train import timed_observe
    calls = []

    class Executor:
        def __init__(self, kind): self.kind = kind
        def submit(self, function, *args):
            future = Future(); calls.append((self.kind, function, args, future)); return future

    monkeypatch.setattr(module, 'numpy_capable', lambda spec, signatures: spec['id'] != 'unknown')
    pool = object.__new__(FlexiblePool)
    pool.signatures = ('verified',)
    pool.pools = {True: Executor(True), False: Executor(False)}
    pool.router = BoundedRouter({True: 1, False: 1}, pool._dispatch)
    own = {'id': 'own'}; known = {'id': 'known'}; unknown = {'id': 'unknown'}
    a = pool.submit(module.duel, own, known, 10, 2)
    b = pool.submit(timed_observe, own, known, 10, 2)
    c = pool.submit(module.duel, own, unknown, 10, 2)
    assert calls[0][:2] == (True, module.execute)
    assert calls[0][2][-2:] == (pool.signatures, False)
    assert calls[1][:2] == (False, timed_observe)
    calls[1][3].set_result('helper')
    assert calls[2][:2] == (False, module.duel)
    calls[2][3].set_result('unknown'); calls[0][3].set_result('pure')
    pool.router.close()
    assert [a.result(), b.result(), c.result()] == ['pure', 'helper', 'unknown']
    with pytest.raises(ValueError):
        pool.submit(lambda: None, own, known, 10, 2)
