"""After current simulations finish, compare pre-import thread limits.

The waiting controller imports only the standard library. Each trial is a new
Python process, so its NumPy thread environment is set before any NumPy import.
Uses the unchanged official match runner; does not select or train a policy.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
from importlib.metadata import distributions
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
THREAD_KEYS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')
SCENARIOS=(('default_w3','default',3),('single_w3','single',3),('single_w16','single',16))


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def relative(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    data=json.dumps(value,indent=2,ensure_ascii=False)
    for attempt in range(10):
        try:
            temporary.write_text(data,encoding='utf-8');temporary.replace(path);return
        except PermissionError as error:
            if getattr(error,'winerror',None) not in (5,32,33) or attempt==9: raise
            time.sleep(min(.02*2**attempt,.25))


def verify(plan):
    for table in ('source_sha256','input_sha256'):
        for name,digest in plan[table].items():
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT) or sha(path)!=digest:
                raise ValueError(f'Changed frozen file: {name}')


def environment():
    return dict(python=sys.version,windows=list(sys.getwindowsversion()),
        packages=[list(x) for x in sorted((d.metadata['Name'],d.version)
                  for d in distributions() if d.metadata['Name'])])


def process_live(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        if ctypes.get_last_error()==87:return False
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        code=wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
        return code.value==259
    finally:kernel.CloseHandle(handle)


class PerformanceInfo(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
        ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable',
         'SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[
        (k,wintypes.DWORD) for k in ('HandleCount','ProcessCount','ThreadCount')]


class MemoryCounters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t)
        for k in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                  'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]


def resources():
    api=ctypes.WinDLL('psapi',use_last_error=True)
    api.GetPerformanceInfo.argtypes=[ctypes.POINTER(PerformanceInfo),wintypes.DWORD]
    api.GetPerformanceInfo.restype=wintypes.BOOL
    info=PerformanceInfo();info.cb=ctypes.sizeof(info)
    if not api.GetPerformanceInfo(ctypes.byref(info),info.cb):raise ctypes.WinError(ctypes.get_last_error())
    return dict(commit_gib=info.CommitTotal*info.PageSize/2**30,
        commit_headroom_gib=(info.CommitLimit-info.CommitTotal)*info.PageSize/2**30,
        available_memory_gib=info.PhysicalAvailable*info.PageSize/2**30)


def own_memory():
    api=ctypes.WinDLL('psapi',use_last_error=True);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]
    api.GetProcessMemoryInfo.restype=wintypes.BOOL
    counters=MemoryCounters();counters.cb=ctypes.sizeof(counters)
    if not api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(pid=os.getpid(),private_gib=counters.PrivateUsage/2**30,
                working_set_gib=counters.WorkingSetSize/2**30)


class Sampler:
    def __enter__(self):
        self.rows=[resources()];self.stop=threading.Event();self.error=None
        def sample():
            try:
                while not self.stop.wait(.5):self.rows.append(resources())
            except Exception as error:self.error=error
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join()
        if self.error is not None:raise self.error
        self.rows.append(resources())
        self.result=dict(peak_system_commit_gib=max(r['commit_gib'] for r in self.rows),
            peak_commit_increase_gib=max(r['commit_gib'] for r in self.rows)-self.rows[0]['commit_gib'],
            minimum_commit_headroom_gib=min(r['commit_headroom_gib'] for r in self.rows),
            minimum_available_memory_gib=min(r['available_memory_gib'] for r in self.rows))


def worker_initialize(barrier,queue,own,neural):
    # Import NumPy via this module BEFORE the historical initializer runs.
    from tools.league_matches import Actor,initialize_worker
    initialize_worker();Actor(own);Actor(neural)
    import torch
    queue.put(dict(**own_memory(),torch_threads=torch.get_num_threads(),
        thread_environment={key:os.environ.get(key) for key in THREAD_KEYS}))
    barrier.wait(timeout=180)


def ready():return os.getpid()


def trial_environment(base,mode):
    result=dict(base)
    if mode not in ('default','single'):raise ValueError('Unknown thread mode')
    for key in THREAD_KEYS:
        if mode=='single':result[key]='1'
        else:result.pop(key,None)
    return result


def comparable(results):
    return [dict(own=r['own'],foe=r['foe'],band=r['band'],summary=r['summary'],episodes=r['episodes']) for r in results]


def trial(plan_path,out,mode,workers):
    plan=read(plan_path);verify(plan)
    if environment()!=plan['environment']:raise ValueError('Trial environment changed')
    expected=trial_environment({},mode)
    if any(os.environ.get(k)!=expected.get(k) for k in THREAD_KEYS):
        raise ValueError('Thread environment must be set before Python starts')
    for name in ('runtime.json','workers.json'):
        path=out/name
        if path.exists():
            rows=[read(path)] if name=='runtime.json' else read(path)
            if any(process_live(row['pid']) for row in rows):raise ValueError('Prior trial process still live')
    write(out/'runtime.json',dict(pid=os.getpid(),mode=mode,workers=workers,started_at=datetime.now(timezone.utc).isoformat()))
    started=time.perf_counter()
    with Sampler() as sampler:
        from tools.league_matches import duel,ROOT as imported_root
        if imported_root.resolve()!=ROOT:raise ValueError('Wrong project import root')
        imported=time.perf_counter()
        context=mp.get_context('spawn');barrier=context.Barrier(workers);queue=context.Queue()
        try:
            with ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=worker_initialize,
                    initargs=(barrier,queue,plan['jobs'][0]['own'],plan['neural_warmup'])) as pool:
                warm=[pool.submit(ready) for _ in range(workers)]
                memories=[queue.get(timeout=180) for _ in range(workers)]
                if len({m['pid'] for m in memories})!=workers:raise ValueError('Duplicate initialized worker')
                write(out/'workers.json',memories)
                for f in warm:f.result()
                warmed=time.perf_counter()
                futures=[pool.submit(duel,job['own'],job['foe'],job['band'],job['n']) for job in plan['jobs']]
                results=[future.result() for future in futures]
                match_seconds=time.perf_counter()-warmed
        finally:queue.close();queue.join_thread()
    record=dict(request=dict(plan_sha256=sha(plan_path),mode=mode,workers=workers),
        import_seconds=imported-started,startup_seconds=warmed-imported,
        match_seconds=match_seconds,total_seconds=time.perf_counter()-started,
        games=sum(len(r['episodes']) for r in results),resources=sampler.result,
        initialized_workers=memories,results=results)
    verify(plan);write(out/'result.json',record)
    return record


def decision(records):
    grouped={name:[r for r in records if r['scenario']==name] for name,_,_ in SCENARIOS}
    if any(len(rows)!=2 for rows in grouped.values()):raise ValueError('Two reversed-order repeats required')
    reference=comparable(records[0]['results'])
    if any(comparable(r['results'])!=reference for r in records):raise ValueError('Thread setting changed episode records')
    seconds={name:median(r['match_seconds'] for r in rows) for name,rows in grouped.items()}
    private={name:median(sum(w['private_gib'] for w in r['initialized_workers'])/len(r['initialized_workers'])
                        for r in rows) for name,rows in grouped.items()}
    eligible={workers:seconds[name] for name,mode,workers in SCENARIOS if mode=='single' and all(
        r['resources']['minimum_commit_headroom_gib']>=4 and r['resources']['minimum_available_memory_gib']>=2
        for r in grouped[name])}
    selected=min((w for w,t in eligible.items() if t<=1.03*min(eligible.values())),default=None)
    return dict(status='benchmark_complete',exact_episode_equality=True,median_match_seconds=seconds,
        median_initialized_worker_private_gib=private,recommended_workers=selected,
        single_thread_private_reduction_gib_at_three=private['default_w3']-private['single_w3'],
        matched_three_worker_speed_ratio=seconds['default_w3']/seconds['single_w3'],
        preimport_thread_environment={k:'1' for k in THREAD_KEYS},
        rule='Identical episodes; fewest single-thread workers within 3% of fastest, with >=4 GiB commit headroom and >=2 GiB physical memory free in both repeats.',
        scope='Fixed 112 games repeated across execution settings, not independent policy-performance evidence. No 16-worker default-thread speed comparison; OS/background activity can affect system memory and timing.')


def freeze(out,search,audit,after_pids):
    if len(after_pids)!=2 or after_pids!=[read(folder/'runtime.json')['pid'] for folder in (search,audit)]:
        raise ValueError('Wait PIDs must match the actual predecessor runtime records')
    path=out/'plan.json'
    if path.exists():
        plan=read(path);verify(plan)
        if (plan['search'],plan['audit'],plan['after_pids'],plan['environment'])!=(relative(search),relative(audit),after_pids,environment()):
            raise ValueError('Changed queued benchmark configuration')
        return plan
    old=read(search/'plan.json');verify(old)
    names=('ace','evader','circler','refine_2201','ddqn_s0','ddqn_s1','round_05')
    foes=[next(x for x in old['opponents'] if x['id']==name) for name in names]
    sources=dict(old['source_sha256']);sources['tools/league_thread_benchmark.py']=sha(__file__)
    inputs=dict(old['input_sha256']);inputs[relative(search/'plan.json')]=sha(search/'plan.json')
    plan=dict(search=relative(search),audit=relative(audit),after_pids=after_pids,
        jobs=[dict(own=old['anchor'],foe=foe,band=170000000+block*100,n=4) for block in range(4) for foe in foes],
        neural_warmup=next(x for x in foes if x['id']=='ddqn_s1'),
        scenarios=[list(x) for x in SCENARIOS],repeats=2,games_per_trial=112,
        environment=environment(),source_sha256=sources,input_sha256=inputs,
        scope='Wait for both training and its conditional audit, then test execution settings with fixed archived models; no new policy is trained or selected.')
    write(path,plan)
    for name in sources:
        target=out/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    return plan


def run(out,search,audit,after_pids):
    out,search,audit=out.resolve(),search.resolve(),audit.resolve()
    if (out/'runtime.json').exists() and process_live(read(out/'runtime.json')['pid']):raise ValueError('Benchmark already live')
    plan=freeze(out,search,audit,after_pids)
    if (out/'completion.json').exists():return read(out/'completion.json')
    write(out/'runtime.json',dict(pid=os.getpid(),after_pids=after_pids,simulation_workers_while_waiting=0,started_at=datetime.now(timezone.utc).isoformat()))
    write(out/'progress.json',dict(stage='waiting_for_training_and_audit',after_pids=after_pids))
    while any(process_live(pid) for pid in after_pids):time.sleep(45)
    for folder in (search,audit):
        if not (folder/'completion.json').exists():
            result=dict(status='dependency_incomplete',matches_started=False,folder=relative(folder))
            write(out/'completion.json',result);return result
    if read(search/'completion.json')['status'] not in ('evaluation_passed','evaluation_failed','no_development_improvement'):
        result=dict(status='dependency_incomplete',matches_started=False,folder=relative(search))
        write(out/'completion.json',result);return result
    if read(audit/'completion.json')['status'] not in ('audit_complete','not_eligible'):
        result=dict(status='dependency_incomplete',matches_started=False,folder=relative(audit))
        write(out/'completion.json',result);return result
    verify(plan)
    if environment()!=plan['environment']:raise ValueError('Queued benchmark environment changed')
    records=[]
    for repeat,order in enumerate((SCENARIOS,tuple(reversed(SCENARIOS)))):
        for name,mode,workers in order:
            folder=out/f'r{repeat}_{name}';result_path=folder/'result.json'
            if not result_path.exists():
                reserve=4+(1.2 if mode=='default' else .6)*(workers+1)
                if resources()['commit_headroom_gib']<reserve:
                    raise RuntimeError(f'Need {reserve:.1f} GiB commit headroom before this trial')
                write(out/'progress.json',dict(stage='benchmark',repeat=repeat,scenario=name))
                folder.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,'-X','utf8','-u','-m','tools.league_thread_benchmark','--trial',
                    '--out',str(folder),'--plan',str(out/'plan.json'),'--mode',mode,'--workers',str(workers)]
                with (folder/'stdout.log').open('w',encoding='utf-8') as stdout,(folder/'stderr.log').open('w',encoding='utf-8') as stderr:
                    subprocess.run(command,cwd=ROOT,env=trial_environment(os.environ,mode),stdout=stdout,stderr=stderr,
                        creationflags=0x08000000,check=True)
            record=read(result_path)
            if record['request']!=dict(plan_sha256=sha(out/'plan.json'),mode=mode,workers=workers):raise ValueError('Changed trial request')
            records.append(dict(**record,scenario=name,repeat=repeat))
    result=decision(records);verify(plan)
    write(out/'completion.json',result);write(out/'progress.json',dict(stage='complete'))
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--search',type=Path,default=Path('runs/league_repair_20261007'))
    parser.add_argument('--audit',type=Path,default=Path('runs/league_temporal_audit_v3_20261007'))
    parser.add_argument('--after-pids',type=int,nargs='+')
    parser.add_argument('--trial',action='store_true')
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--mode',choices=('default','single'))
    parser.add_argument('--workers',type=int)
    args=parser.parse_args()
    if not args.trial and (not args.after_pids or len(args.after_pids)!=2):
        parser.error('Supply the actual training and audit process IDs with --after-pids')
    try:
        if args.trial:trial(args.plan,args.out,args.mode,args.workers)
        else:run(args.out,args.search,args.audit,args.after_pids)
    except Exception:
        write(args.out/'failure.json',dict(traceback=traceback.format_exc()))
        raise
