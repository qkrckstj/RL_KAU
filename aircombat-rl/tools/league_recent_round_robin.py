"""Balanced role-level cross-play among preserved and newly learned policies."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import os,shutil,time,traceback
from tools import league_thread_benchmark as io
from tools.league_matches import duel,initialize_worker,summary
from tools.league_sampled_numpy_qualify import validate_job_result


def analyze(plan,rows):
    groups=list(plan['roles']);matrix={a:{b:dict(wins=0,draws=0,losses=0,games=0) for b in groups if b!=a} for a in groups}
    mapping={s['id']:role for role,specs in plan['roles'].items() for s in specs}
    for r in rows:
        a,b=mapping[r['own']],mapping[r['foe']];s=r['summary']
        for own,foe,w,l in [(a,b,s['wins'],s['losses']),(b,a,s['losses'],s['wins'])]:
            cell=matrix[own][foe];cell['wins']+=w;cell['losses']+=l;cell['draws']+=s['draws'];cell['games']+=len(r['episodes'])
    for a in groups:
        for b,cell in matrix[a].items():
            assert cell['games']==len(plan['roles'][a])*len(plan['roles'][b])*plan['n']
            cell['win_rate']=cell['wins']/cell['games'];cell['score']=(cell['wins']+.5*cell['draws'])/cell['games']
    aggregates={a:dict(equal_role_win_rate=sum(c['win_rate'] for c in row.values())/len(row),equal_role_score=sum(c['score'] for c in row.values())/len(row),worst_role_score=min(c['score'] for c in row.values())) for a,row in matrix.items()}
    return dict(matrix=matrix,aggregates=aggregates,scope=plan['scope'],policy_promoted=False)


def run(source,out):
    if out.exists():raise FileExistsError(out)
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']):raise RuntimeError('PPO learning still live')
    old=io.read(source/'plan.json');io.verify(old);done=io.read(source/'completion.json');raw=io.read(source/'raw_hold_analysis.json')
    if done['status']!='hold_pilots_complete' or raw['status']!='hold_raw_verified':raise ValueError('Verified completed pilots required')
    for p,h in raw['input_sha256'].items():
        if io.sha(io.ROOT/p)!=h:raise ValueError('Changed learner evidence')
    probe=io.read(io.ROOT/old['source']/'plan.json');portfolio_path=io.ROOT/probe['source'];portfolio=io.read(portfolio_path/'plan.json');portfolio_done=io.read(portfolio_path/'completion.json');geometry=io.read(io.ROOT/portfolio['source']/'plan.json')
    roles=dict(baseline=[r['spec'] for r in old['warm_starts']['control']['replicas']])
    for r in done['searches']:roles[f"{r['arm']}_s{r['seed']}"]=[a['spec'] for a in r['history'][-1]['candidate']['replicas']]
    for r in portfolio_done['searches']:roles[f"portfolio_s{r['seed']}"]=r['selected']['specs']
    for name,specs in geometry['roles'].items():
        if name.startswith('focal_'):roles[name]=specs
    roles['history']=[next(s for s in old['opponents'] if s['id']==f'history_s6000_t1048576_a{a}') for a in [4900,4901]]
    roles['teacher']=[old['teacher']]
    for name in ['ace','temporal_extend_right','temporal_weave_right','evader']:roles[name]=[next(s for s in old['opponents'] if s['id']==name)]
    all_specs=[s for specs in roles.values() for s in specs]
    assert len({s['id'] for s in all_specs})==len(all_specs)
    expanded=list(old['opponents'])
    for s in all_specs:
        previous=next((f for f in expanded if f['id']==s['id']),None)
        if previous is None:expanded.append(s)
        elif previous!=s:raise ValueError('Policy ID collision')
    band=94000000;n=4;names=list(roles)
    jobs=[dict(own=a,foe=b,band=band,n=n,traced=False) for i,role in enumerate(names) for other in names[i+1:] for a in roles[role] for b in roles[other]]
    sources=dict(old['source_sha256']);inputs=dict(old['input_sha256']);sources[io.relative(Path(__file__).resolve())]=io.sha(Path(__file__).resolve())
    for p in [source/'plan.json',source/'completion.json',source/'raw_hold_analysis.json',portfolio_path/'plan.json',portfolio_path/'completion.json']:inputs[io.relative(p)]=io.sha(p)
    for s in all_specs:
        if s['kind']=='submission':
            for p in (io.ROOT/s['design']).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:inputs[io.relative(p)]=io.sha(p)
    claim=io.ROOT/f'runs/sampled_assessment_reservation_{band}.json'
    if claim.exists():raise FileExistsError('Round-robin conditions reserved')
    plan=dict(source=io.relative(source),roles=roles,expanded_training_archive=expanded,previous_archive_size=len(old['opponents']),band=band,n=n,jobs=jobs,workers=8,source_sha256=sources,input_sha256=inputs,environment=io.environment(),scope='Exploratory cross-play: all four new PPO finals included regardless of rank, both prior focal experts and both CEM gate searches, parent/history/teacher and four basic weaknesses. Every unordered role pair, every action-RNG combination, two fresh ICs and both physical seats. Official symmetric kill/died inversion counts the same battle for the opponent; mutual/timeout stay draws. Equal role weighting avoids giving two-replica roles double aggregate weight. New learned policies were absent from this PPO training archive. Not a broad tournament/final/heldout claim or automatic promotion.')
    io.verify(plan);io.write(claim,dict(run=io.relative(out),start=band,stop_exclusive=band+n//2));io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    if min(io.resources()[k] for k in ['commit_headroom_gib','available_memory_gib'])<4.5:raise MemoryError('Startup memory')
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    started=time.perf_counter();rows=[None]*len(jobs)
    try:
        with ProcessPoolExecutor(max_workers=8,initializer=initialize_worker) as pool:
            futures={pool.submit(duel,j['own'],j['foe'],band,n):i for i,j in enumerate(jobs)}
            for count,future in enumerate(as_completed(futures),1):
                i=futures[future];r=future.result();validate_job_result(jobs[i],r);assert r['summary']==summary(r['episodes']);rows[i]=r;io.write(out/f'matches/match_{i:04d}.json',dict(job=jobs[i],result=r))
                if count%24==0:print('recent-policy cross-play',count,len(jobs),flush=True)
                if min(io.resources()[k] for k in ['commit_headroom_gib','available_memory_gib'])<1.8:raise MemoryError('Runtime memory')
        io.verify(plan);analysis=analyze(plan,rows);io.write(out/'analysis.json',analysis)
        io.write(out/'completion.json',dict(status='recent_round_robin_complete',games=sum(len(r['episodes']) for r in rows),wall_seconds=time.perf_counter()-started,expanded_archive_size=len(expanded),policy_promoted=False,final_opened=False,heldout_opened=False))
    except BaseException:io.write(out/'failure.json',dict(traceback=traceback.format_exc(),resources=io.resources()));raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source.resolve(),a.out.resolve())
