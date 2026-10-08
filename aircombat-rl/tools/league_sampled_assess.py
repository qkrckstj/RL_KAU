"""Fresh paired assessment of a development-frozen sampled PPO nominee.

All archived opponents remain; final/heldout bands are left unopened. This
diagnoses transfer to new initial conditions and does not promote a policy.
"""
from argparse import ArgumentParser
from concurrent.futures import as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from tools import league_thread_benchmark as io
from tools.league_sampled_numpy_qualify import validate_job_result
from tools.league_tournament_metrics import code_groups, group_weights, profile
from tools.league_matches import duel
from tools.league_split_pool import SplitPool


def vectors(records, opponents, band, n, weights):
    if n<=0 or n%2 or not weights or any(w<=0 for w in weights.values()) or not np.isclose(sum(weights.values()),1):raise ValueError('Complete seat pairs and normalized positive weights required')
    expected={(band+i,seat) for i in range(n//2) for seat in ('red','blue')}
    if set(weights)!={s['id'] for s in opponents}:raise ValueError('Wrong weights')
    values=[]
    for rows in records:
        if len(rows)!=len(opponents) or {r['foe'] for r in rows}!=set(weights):raise ValueError('Wrong panel')
        value=np.zeros(n//2)
        for row in rows:
            episodes={(e['seed'],e['seat']):e for e in row['episodes']}
            if len(episodes)!=n or set(episodes)!=expected:raise ValueError('Unpaired episodes')
            value+=weights[row['foe']]*np.array([sum(e['outcome']=='kill' for k,e in episodes.items() if k[0]==band+i)/2 for i in range(n//2)])
        values.append(value)
    return np.array(values)


def gains(new, old, opponents, band, n, weights):
    a,b=(vectors(rows,opponents,band,n,weights) for rows in (new,old))
    if a.shape[0]!=2 or b.shape[0] not in (1,2):raise ValueError('Expected two candidate action replicas')
    differences=a-b
    rng=np.random.default_rng(815);ics=rng.integers(n//2,size=(10000,n//2))
    replicas=rng.integers(2,size=(10000,2))
    conditional=differences.mean(axis=0)[ics].mean(axis=1)
    crossed=differences[replicas[:,:,None],ics[:,None,:]].mean(axis=(1,2))
    return dict(mean=float(differences.mean()),per_action_replica_mean=differences.mean(axis=1).tolist(),
        ic_cluster_ci95=np.quantile(conditional,[.025,.975]).tolist(),
        crossed_ic_action_ci95=np.quantile(crossed,[.025,.975]).tolist(),initial_conditions=n//2,
        scope='Conditional on this frozen archive and two fixed action RNGs; IC clusters retain both seats/all foes. Neither independent retraining nor uncertainty over arbitrary future entrants.')


def freeze(source, qualification, out):
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Set thread limits first')
    for path in (source,qualification):
        if io.process_live(io.read(path/'runtime.json')['pid']):raise RuntimeError('Predecessor still live')
    completed=io.read(source/'completion.json');qualified=io.read(qualification/'completion.json')
    if completed['status']!='sampled_recovery_continuations_complete':raise ValueError('Incomplete learning')
    if qualified['status']!='sampled_numpy_execution_qualified' or not qualified['exact_episode_and_trace_equality']:raise ValueError('Missing loader qualification')
    old=io.read(source/'plan.json');qp=io.read(qualification/'plan.json');io.verify(old);io.verify(qp)
    checks_path=io.ROOT/'runs/sampled_assessment_analysis_checks_20261007.json'
    checks=io.read(checks_path)
    if checks['status']!='passed' or checks['source_sha256']!=io.sha(Path(__file__)):raise ValueError('Matching aggregation checks required')
    if qp['source']!=io.relative(source):raise ValueError('Qualification refers to another source')
    if (out/'plan.json').exists():
        plan=io.read(out/'plan.json');io.verify(plan)
        if plan['environment']!=io.environment() or plan['source']!=io.relative(source) or plan['qualification']!=io.relative(qualification):raise ValueError('Changed assessment resume')
        return plan
    if out.exists():raise FileExistsError('Use new output')
    nominated=max(completed['searches'],key=lambda s:tuple(s['selected']['rank']))
    chosen=nominated['selected']
    opponents=list(old['opponents'])
    for search in completed['searches']:
        if search is nominated:continue
        for replica in search['selected']['replicas']:
            spec=replica['spec'];previous=next((s for s in opponents if s['id']==spec['id']),None)
            if previous is not None and previous!=spec:raise ValueError('Opponent ID collision')
            if previous is None:opponents.append(spec)
    roles=dict(candidate=[r['spec'] for r in chosen['replicas']],warm=[r['spec'] for r in old['fallback_baseline']['replicas']],early=[old['early_greedy_comparator']],teacher=[old['teacher']])
    if any([r['action_seed'] for r in item['replicas']]!=[4900,4901] for item in (chosen,old['fallback_baseline'])):raise ValueError('Changed fixed action seeds')
    band,n=81000000,16
    jobs=[dict(own=spec,foe=foe,band=band,n=n,traced=False) for role in roles.values() for spec in role for foe in opponents]
    if len({(j['own']['id'],j['foe']['id']) for j in jobs})!=len(jobs):raise ValueError('Duplicate assessment policies')
    sources=dict(old['source_sha256']);sources.update(qp['source_sha256'])
    sources['tools/league_sampled_assess.py']=io.sha(Path(__file__))
    inputs=dict(old['input_sha256']);inputs.update(qp['input_sha256'])
    for path in [source/'plan.json',source/'completion.json',qualification/'plan.json',qualification/'completion.json',checks_path]:
        inputs[io.relative(path)]=io.sha(path)
    for spec in [s for r in roles.values() for s in r]+opponents:
        if spec['kind']=='submission':
            for path in list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]:inputs[io.relative(path)]=io.sha(path)
    claim=io.ROOT/f'runs/sampled_assessment_reservation_{band}.json'
    if claim.exists():raise FileExistsError('Fresh assessment band already reserved')
    plan=dict(source=io.relative(source),qualification=io.relative(qualification),chosen=chosen,roles=roles,
        opponents=opponents,groups=code_groups(opponents),band=band,n=n,jobs=jobs,
        workers=8,neural_workers=2,approved_signatures=qualified['approved_signatures'],
        pure_warmup=qp['pure_warmup'],neural_warmup=qp['neural_warmup'],environment=io.environment(),
        source_sha256=sources,input_sha256=inputs,
        rule='Nominee fixed solely by previous two-replica development rank before fresh games. All96 archive foes plus other continuation nominees, eight new ICs/both seats, both action RNGs equally weighted; no best action-seed choice.',
        next_rule='Report aggregate and weakness changes versus warm8M, preserved early2M greedy, and CEM teacher. If gains do not transfer or extend-right remains weak, use this now-consumed panel for a new explicitly frozen repair strategy rather than blindly extending current learner.',
        scope='Fresh initial-condition assessment of known opponents. Shared-prefix continuations are not from-scratch replication. No final/heldout test, promotion, or GitHub publication.')
    io.verify(plan);io.write(claim,dict(run=io.relative(out),band=band,stop_exclusive=band+n//2));io.write(out/'plan.json',plan)
    for name in sources:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,dest)
    return plan


def analyze(plan, rows):
    by_own={s['id']:[r for r in rows if r['own']==s['id']] for specs in plan['roles'].values() for s in specs}
    roles={role:[by_own[s['id']] for s in specs] for role,specs in plan['roles'].items()}
    stats={}
    for role,replicas in roles.items():
        profiles=[profile(r,plan['groups']) for r in replicas]
        stats[role]=dict(profiles=profiles,mean_win_rate=float(np.mean([p['mean_win_rate'] for p in profiles])),
            group_balanced_win_rate=float(np.mean([p['group_balanced_win_rate'] for p in profiles])),
            conservative_lower_quarter=min(p['lower_quarter_score'] for p in profiles),worst_score=min(p['worst_score'] for p in profiles),
            opponents={foe['id']:dict(wins=sum(r['summary']['wins'] for rs in replicas for r in rs if r['foe']==foe['id']),
                draws=sum(r['summary']['draws'] for rs in replicas for r in rs if r['foe']==foe['id']),
                losses=sum(r['summary']['losses'] for rs in replicas for r in rs if r['foe']==foe['id']),games=plan['n']*len(replicas)) for foe in plan['opponents']})
    weights=dict(uniform={s['id']:1/len(plan['opponents']) for s in plan['opponents']},code_group=group_weights(plan['groups']))
    comparisons={role:{label:gains(roles['candidate'],roles[role],plan['opponents'],plan['band'],plan['n'],w) for label,w in weights.items()} for role in ('warm','early','teacher')}
    return dict(roles=stats,comparisons=comparisons,policy_promoted=False,final_opened=False,heldout_opened=False,scope=plan['scope'])


def run(source,qualification,out):
    if (out/'runtime.json').exists() and io.process_live(io.read(out/'runtime.json')['pid']):raise RuntimeError('Assessment already live')
    plan=freeze(source,qualification,out)
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),workers=plan['workers']))
    if min(io.resources()[k] for k in ('commit_headroom_gib','available_memory_gib'))<4.5:raise MemoryError('Insufficient eight-worker assessment startup headroom')
    started=time.perf_counter();rows=[None]*len(plan['jobs'])
    with io.Sampler() as sampler,SplitPool(plan['workers'],plan['neural_workers'],plan['approved_signatures'],plan['pure_warmup'],plan['neural_warmup']) as pool:
        io.write(out/'initialized_workers.json',dict(workers=pool.initialized_workers,resources=io.resources()))
        if io.resources()['commit_headroom_gib']<3.0:raise MemoryError('Insufficient initialized assessment headroom')
        futures={}
        for i,job in enumerate(plan['jobs']):
            path=out/f'matches/match_{i:04d}.json'
            if path.exists():
                saved=io.read(path)
                if saved['job']!=job:raise ValueError('Changed saved job')
                result=saved['result']
                validate_job_result(job,result);rows[i]=result
            else:
                resources=io.resources()
                if resources['commit_headroom_gib']<1.8 or resources['available_memory_gib']<1.5:raise MemoryError('Assessment resource floor')
                futures[pool.submit(duel,job['own'],job['foe'],job['band'],job['n'])]=(i,job,path)
        for future in as_completed(futures):
            i,job,path=futures[future];result=future.result()
            validate_job_result(job,result);io.write(path,dict(job=job,result=result));rows[i]=result
            done=sum(r is not None for r in rows)
            if done%24==0:print(f'assessment: {done}/{len(plan["jobs"])} match batches',flush=True)
            resources=io.resources()
            if resources['commit_headroom_gib']<1.8 or resources['available_memory_gib']<1.5:
                for pending in futures:pending.cancel()
                raise MemoryError('Assessment resource floor')
    io.verify(plan);analysis=analyze(plan,rows);io.write(out/'analysis.json',analysis)
    io.write(out/'completion.json',dict(status='sampled_fresh_assessment_complete',games=sum(len(r['episodes']) for r in rows),
        wall_seconds=time.perf_counter()-started,minimum_commit_headroom_gib=min(r['commit_headroom_gib'] for r in sampler.rows),
        **{k:analysis[k] for k in ('policy_promoted','final_opened','heldout_opened','scope')},completed_at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--qualification',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:run(args.source.resolve(),args.qualification.resolve(),args.out.resolve())
    except BaseException:
        io.write(args.out/f'failure_{time.time_ns()}.json',dict(error=traceback.format_exc(),at=datetime.now(timezone.utc).isoformat()));raise
