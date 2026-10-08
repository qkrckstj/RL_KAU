"""Resume an interrupted frozen assessment, preserving and validating saved jobs."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import ctypes
from ctypes import wintypes
from tools import league_thread_benchmark as io
from tools import league_sampled_assess as engine
from tools import league_prioritized_ppo_assess as original


def old_worker_state(pid):
    if not io.process_live(pid):
        return dict(pid=pid, state='absent')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        size = wintypes.DWORD(32768); name = ctypes.create_unicode_buffer(size.value)
        if not kernel.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        # Observed PID reuse by Windows console host. Never stop the unrelated process.
        assert Path(name.value) == Path(os.environ['WINDIR'])/'System32/conhost.exe', 'Old worker PID still live or identity unresolved'
        return dict(pid=pid, state='reused_by_unrelated_console_host', image=name.value)
    finally:
        kernel.CloseHandle(handle)


def checked_existing_plan(source, qualification, out):
    plan = io.read(out/'plan.json'); io.verify(plan)
    assert plan['source'] == io.relative(source)
    assert plan['qualification'] == io.relative(qualification)
    assert plan['environment'] == io.environment()
    assert all(os.environ.get(k) == '1' for k in io.THREAD_KEYS)
    assert not (out/'completion.json').exists()
    runtime = io.read(out/'runtime.json')
    assert not io.process_live(runtime['pid']), 'Controller still live'
    workers = io.read(out/'initialized_workers.json')
    worker_states = [old_worker_state(w['pid']) for w in workers['workers']]
    assert min(io.resources()[k] for k in ('commit_headroom_gib', 'available_memory_gib')) >= 6
    saved = {}
    for i, job in enumerate(plan['jobs']):
        f = out/f'matches/match_{i:04d}.json'
        if f.exists():
            d = io.read(f)
            assert d['job'] == job
            original.validate_job_result(job, d['result'])
            saved[io.relative(f)] = io.sha(f)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    audit = out/'resume_attempts'/stamp
    audit.mkdir(parents=True)
    for name in ['runtime.json', 'initialized_workers.json']:
        shutil.copyfile(out/name, audit/name)
    io.write(audit/'manifest.json', dict(reason='Controller and workers absent; original session unavailable; no terminal report. Cause unknown.',
        at=datetime.now(timezone.utc).isoformat(), resources=io.resources(), worker_states=worker_states, saved_batches=len(saved),
        remaining_batches=len(plan['jobs'])-len(saved), saved_sha256=saved,
        plan_sha256=io.sha(out/'plan.json'), resume_source_sha256=io.sha(Path(__file__)),
        original_source_sha256=io.sha(Path(original.__file__)),
        timing_scope='Completion wall time describes this resumed invocation, including validation of reused results; not total first-attempt plus resumed wall time.'))
    print(f'resume verified: {len(saved)} saved batches, {len(plan["jobs"])-len(saved)} remaining', flush=True)
    return plan


if __name__ == '__main__':
    p = ArgumentParser(); p.add_argument('out', type=Path); a = p.parse_args()
    out = a.out.resolve(); plan = io.read(out/'plan.json')
    engine.freeze = checked_existing_plan; engine.analyze = original.metrics.analyze
    engine.run(io.ROOT/plan['source'], io.ROOT/plan['qualification'], out)
    original.verify_results(out)
    for manifest in (out/'resume_attempts').glob('*/manifest.json'):
        for path, digest in io.read(manifest)['saved_sha256'].items():
            assert io.sha(io.ROOT/path) == digest, 'Saved result changed during resume'
    print('resumed saved-result hashes verified', flush=True)
