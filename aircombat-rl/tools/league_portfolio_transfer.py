"""Fresh-condition assessment of the development-selected portfolio gate."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import os,shutil,time,traceback
from tools import league_thread_benchmark as io
from tools.league_matches import duel,initialize_worker
from tools.league_sampled_numpy_qualify import validate_job_result
from tools.league_sampled_assess import gains
from tools.league_tournament_metrics import code_groups,group_weights,profile


def analyze(plan,rows):
    records={role:[[r for r in rows if r['own']==s['id']] for s in specs] for role,specs in plan['roles'].items()}
    stats={}
    for role,replicas in records.items():
        profiles=[profile(r,plan['groups']) for r in replicas]
        stats[role]=dict(profiles=profiles,mean_win_rate=sum(p['mean_win_rate'] for p in profiles)/2,group_balanced_win_rate=sum(p['group_balanced_win_rate'] for p in profiles)/2,opponents={f['id']:{k:sum(r['summary'][k] for part in replicas for r in part if r['foe']==f['id']) for k in ['wins','draws','losses']} for f in plan['opponents']})
    comparisons={}
    trained=set(plan['gate_training_opponents'])
    for label,foes in [('all',plan['opponents']),('gate_training_panel',[s for s in plan['opponents'] if s['id'] in trained]),('outside_gate_training_panel',[s for s in plan['opponents'] if s['id'] not in trained])]:
        ids={s['id'] for s in foes};sub={role:[[r for r in part if r['foe'] in ids] for part in parts] for role,parts in records.items()}
        weights=dict(uniform={s['id']:1/len(foes) for s in foes},code_group=group_weights(code_groups(foes)))
        comparisons[label]={name:gains(sub['candidate'],sub['baseline'],foes,plan['band'],plan['n'],w) for name,w in weights.items()}
    return dict(roles=stats,comparisons=comparisons,scope=plan['scope'],policy_promoted=False,final_opened=False,heldout_opened=False)


def run(source,out):
    if out.exists():raise FileExistsError(out)
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']):raise RuntimeError('Search still live')
    old=io.read(source/'plan.json');io.verify(old);complete=io.read(source/'completion.json');raw=io.read(source/'raw_portfolio_analysis.json')
    if complete['status']!='portfolio_cem_complete' or raw['status']!='portfolio_raw_verified':raise ValueError('Verified completed search required')
    for p,h in raw['input_sha256'].items():
        if io.sha(io.ROOT/p)!=h:raise ValueError('Changed search evidence')
    chosen=max(complete['searches'],key=lambda r:tuple(r['development']['rank']))
    roles=dict(candidate=chosen['selected']['specs'],baseline=[s[0] for s in old['experts']]);band=92000000;n=8
    jobs=[dict(own=s,foe=f,band=band,n=n,traced=False) for specs in roles.values() for s in specs for f in old['opponents']]
    sources=dict(old['source_sha256']);inputs=dict(old['input_sha256'])
    for p in [Path(__file__).resolve(),io.ROOT/'tools/league_sampled_assess.py']:sources[io.relative(p)]=io.sha(p)
    checks=io.ROOT/'runs/sampled_assessment_analysis_checks_20261007.json';check=io.read(checks)
    if check['status']!='passed' or check['source_sha256']!=io.sha(io.ROOT/'tools/league_sampled_assess.py'):raise ValueError('Paired aggregation checks mismatch')
    for p in [source/'plan.json',source/'completion.json',source/'raw_portfolio_analysis.json',checks]:inputs[io.relative(p)]=io.sha(p)
    for specs in roles.values():
        for s in specs:
            for p in (io.ROOT/s['design']).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:inputs[io.relative(p)]=io.sha(p)
    reservation=io.ROOT/f'runs/sampled_assessment_reservation_{band}.json'
    if reservation.exists():raise FileExistsError('Transfer conditions reserved')
    plan=dict(source=io.relative(source),chosen=chosen,roles=roles,opponents=old['opponents'],groups=code_groups(old['opponents']),gate_training_opponents=[s['id'] for s in old['training_opponents']],band=band,n=n,jobs=jobs,workers=8,source_sha256=sources,input_sha256=inputs,environment=io.environment(),scope='Nominee selected only by90M development rank; both CEM repeats reported, only the better nominee transferred. Fresh92M conditions, all104 known archived foes, both seats and two fixed action replicas. Outside-gate-training means unseen by this13-foe gate search only: underlying experts may have trained against them. No genuinely novel-opponent/final/heldout claim, automatic promotion or GitHub upload.')
    io.verify(plan);io.write(reservation,dict(run=io.relative(out),start=band,stop_exclusive=band+n//2));io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    if min(io.resources()[k] for k in ['commit_headroom_gib','available_memory_gib'])<4.5:raise MemoryError('Insufficient startup memory')
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    started=time.perf_counter();rows=[None]*len(jobs)
    try:
        with ProcessPoolExecutor(max_workers=8,initializer=initialize_worker) as pool:
            futures={pool.submit(duel,j['own'],j['foe'],band,n):i for i,j in enumerate(jobs)}
            for count,future in enumerate(as_completed(futures),1):
                i=futures[future];r=future.result();validate_job_result(jobs[i],r);rows[i]=r;io.write(out/f'matches/match_{i:04d}.json',dict(job=jobs[i],result=r))
                if count%24==0:print('portfolio transfer',count,len(jobs),flush=True)
                if min(io.resources()[k] for k in ['commit_headroom_gib','available_memory_gib'])<1.8:raise MemoryError('Transfer memory floor')
        io.verify(plan);analysis=analyze(plan,rows);io.write(out/'analysis.json',analysis)
        io.write(out/'completion.json',dict(status='portfolio_transfer_complete',games=sum(len(r['episodes']) for r in rows),wall_seconds=time.perf_counter()-started,policy_promoted=False,final_opened=False,heldout_opened=False))
    except BaseException:io.write(out/'failure.json',dict(traceback=traceback.format_exc(),resources=io.resources()));raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source.resolve(),a.out.resolve())
