"""Qualify reviewed NumPy policy families after both recovery learners finish.

Runs the same56 official games through ordinary and split loaders. This is an
execution-equivalence check, not model selection or a throughput benchmark.
Existing frozen profiles and the current training controller are untouched.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import traceback
from tools import league_thread_benchmark as io

REVIEWED = {
    '946b1ef594311f2030b2ef3eabc369388602acd25bcedfabb59b4234d33d0cbc',
    'be35d0c784a85dc0960f4933183645d53a896ad995f5ee8b04f68ebae343ca3f',
    '685c96f2a352cd7fb4eb83d64971cce353ead33ebb59435a88211eb5804646ea',
    '16a6bdd600495bd245094c0c336ad6202eb934ccce888842f8d7fbf40f78fc4e',
}


def freeze(source, out):
    if out.exists():
        raise FileExistsError('Use a new qualification output directory')
    runtime = io.read(source/'runtime.json')
    if io.process_live(runtime['pid']):
        raise RuntimeError('Recovery controller is still live; no concurrent simulation pool')
    completed = io.read(source/'completion.json')
    if completed['status'] != 'sampled_recovery_continuations_complete':
        raise ValueError('Both recovery learners must finish successfully first')
    old = io.read(source/'plan.json')
    io.verify(old)
    review_path = io.ROOT/'runs/sampled_numpy_source_review_20261007.json'
    review = io.read(review_path)
    if {r['signature'] for r in review['reviewed']} != REVIEWED:
        raise ValueError('Static review does not match the four inspected source families')
    from tools.league_numpy_matches import code_signature, numpy_capable
    profile_path = io.ROOT/'experiments/league/execution_profile.json'
    profile = io.read(profile_path)
    approved = sorted(set(profile['approved_signatures']) | REVIEWED)
    for row in review['reviewed']:
        spec = row['representative']
        if code_signature(spec) != row['signature']:
            raise ValueError('Reviewed source changed')
        for name,digest in row['source_sha256'].items():
            if io.sha(io.ROOT/spec['design']/name) != digest:
                raise ValueError('Reviewed file changed')
    chosen = max(completed['searches'],key=lambda r:tuple(r['selected']['rank']))['selected']
    sampled = {r['action_seed']:r['spec'] for r in chosen['replicas']}
    # Name the source families explicitly; JSON ordering is not trusted.
    contextual = next(r['representative'] for r in review['reviewed'] if r['signature'].startswith('946b1ef5'))
    pursuit = next(r['representative'] for r in review['reviewed'] if r['signature'].startswith('be35d0c7'))
    own = [contextual,pursuit,chosen['greedy'],sampled[4900]]
    if {code_signature(s) for s in own} != REVIEWED:
        raise ValueError('Qualification own-policy roster misses an inspected family')
    foes = {s['id']:s for s in old['opponents']}
    jobs = []
    for pilot in own:
        for foe in (foes['ace'],foes['temporal_extend_right'],sampled[4901]):
            jobs.append(dict(own=pilot,foe=foe,band=170060000,n=4,
                traced=foe['id']=='temporal_extend_right' or (pilot==sampled[4900] and foe['id']=='ace')))
    jobs += [dict(own=sampled[4900],foe=foes[name],band=170060000,n=4,traced=False)
             for name in ('ddqn_s0','ddqn_s1')]
    unknown = [s for s in old['opponents'] if not numpy_capable(s,approved)]
    if sorted(s['id'] for s in unknown) != ['ddqn_s0','ddqn_s1']:
        raise ValueError('Unexpected policies outside inspected source families')
    for job in jobs:
        job['expected_pure_route'] = all(numpy_capable(s,approved) for s in (job['own'],job['foe']))
    sources,inputs = dict(old['source_sha256']),dict(old['input_sha256'])
    for name in ('tools/league_sampled_numpy_qualify.py','tools/league_split_pool.py',
                 'tools/league_numpy_matches.py','tools/league_damage_trace.py','tools/league_repair_train.py'):
        sources[name]=io.sha(io.ROOT/name)
    paths=[source/'plan.json',source/'completion.json',review_path,profile_path]
    for spec in own+[sampled[4901]]+old['opponents']:
        if spec['kind']=='submission':
            paths+=list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]
    for path in paths:
        inputs[io.relative(path)]=io.sha(path)
    plan=dict(source=io.relative(source),selected=chosen,jobs=jobs,
        approved_candidate_signatures=approved,new_candidate_signatures=sorted(REVIEWED),
        pure_warmup=profile['pure_warmup']+own+[sampled[4901]],neural_warmup=profile['neural_warmup']+own,
        source_sha256=sources,input_sha256=inputs,environment=io.environment(),
        games_per_trial=sum(j['n'] for j in jobs),scenarios=['ordinary2','split4'],
        rule='Exact saved official episodes/summaries and selected full damage traces; pure workers must not import Torch; DQN routing remains ordinary. Runtime commit>=1.8GiB and physical>=1.5GiB.',
        scope='Compatibility band only, no selection/final evidence, no speed ranking or change to previous frozen execution profile.')
    io.verify(plan)
    io.write(out/'plan.json',plan)
    for name in sources:
        target=out/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(io.ROOT/name,target)
    return plan


def validate_job_result(job,result):
    if any(result[k]!=job[k]['id'] for k in ('own','foe')) or result['band']!=job['band']:
        raise AssertionError('Wrong policy or condition in qualification result')
    expected={(seed,seat) for seed in range(job['band'],job['band']+job['n']//2) for seat in ('red','blue')}
    episodes=result['episodes']
    if len(episodes)!=job['n'] or {(e['seed'],e['seat']) for e in episodes}!=expected:
        raise AssertionError('Missing/duplicate qualification seat or initial condition')
    wins=sum(e['outcome']=='kill' for e in episodes)
    losses=sum(e['outcome']=='died' for e in episodes)
    for key,value in dict(wins=wins,losses=losses,draws=len(episodes)-wins-losses).items():
        if result['summary'][key]!=value:
            raise AssertionError('Official outcome summary differs from raw episodes')
    if job['traced']:
        traces=result['damage_traces']
        if len(traces)!=job['n'] or {(e['seed'],e['seat']) for e in traces}!=expected:
            raise AssertionError('Missing/duplicate qualification damage trace')


def trial(plan,out,split):
    from tools.league_split_pool import SplitPool, initialize, ready
    from tools.league_matches import duel
    from tools.league_repair_train import timed_observe
    before=io.resources()
    if min(before[k] for k in ('commit_headroom_gib','available_memory_gib'))<3.7:
        raise MemoryError('Insufficient qualification startup headroom')
    context=mp.get_context('spawn')
    queue=None
    started=time.perf_counter()
    results=[None]*len(plan['jobs'])
    with io.Sampler() as sampler:
        if split:
            pool=SplitPool(4,2,plan['approved_candidate_signatures'],plan['pure_warmup'],plan['neural_warmup'])
        else:
            queue=context.Queue()
            pool=ProcessPoolExecutor(max_workers=2,mp_context=context,initializer=initialize,
                initargs=(context.Barrier(2),queue,plan['neural_warmup'],plan['approved_candidate_signatures'],False))
        try:
            with pool:
                if split:
                    workers=pool.initialized_workers
                else:
                    futures=[pool.submit(ready) for _ in range(2)]
                    workers=[queue.get(timeout=180) for _ in range(2)]
                    for future in futures:future.result()
                io.write(out/'workers.json',workers)
                if io.resources()['commit_headroom_gib']<2.8:
                    raise MemoryError('Insufficient initialized qualification headroom')
                futures={pool.submit(timed_observe if job['traced'] else duel,
                    job['own'],job['foe'],job['band'],job['n']):i for i,job in enumerate(plan['jobs'])}
                try:
                    for future in as_completed(futures):
                        i=futures[future]
                        results[i]=future.result()
                        validate_job_result(plan['jobs'][i],results[i])
                        io.write(out/f'matches/match_{i:03d}.json',dict(job=plan['jobs'][i],result=results[i]))
                        resources=io.resources()
                        if resources['commit_headroom_gib']<1.8 or resources['available_memory_gib']<1.5:
                            raise MemoryError('Qualification resource floor crossed')
                except BaseException:
                    for pending in futures:pending.cancel()
                    raise
        finally:
            if queue is not None:
                queue.close();queue.join_thread()
    if split and (sum(w['pure'] for w in workers)!=2 or any(w['torch_imported'] for w in workers if w['pure'])):
        raise AssertionError('Pure worker initialization/routing failure')
    if sampler.result['minimum_commit_headroom_gib']<1.8 or sampler.result['minimum_available_memory_gib']<1.5:
        raise MemoryError('Sampled resource minimum failed qualification')
    result=dict(scenario='split4' if split else 'ordinary2',results=results,
        workers=workers,resources=sampler.result,wall_seconds=time.perf_counter()-started,
        games=sum(len(r['episodes']) for r in results))
    io.verify(plan)
    io.write(out/'result.json',result)
    return result


def run(source,out):
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):
        raise ValueError('Set thread limits before Python starts')
    plan=freeze(source,out)
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat()))
    original=trial(plan,out/'ordinary2',False)
    split=trial(plan,out/'split4',True)
    def without_timing(rows):
        return [{k:v for k,v in row.items() if k!='elapsed_seconds'} for row in rows]
    if without_timing(original['results'])!=without_timing(split['results']):
        raise AssertionError('Execution backend changed official records or damage traces')
    if any(r['games']!=plan['games_per_trial'] for r in (original,split)):
        raise AssertionError('Incomplete qualification games')
    io.verify(plan)
    io.write(out/'completion.json',dict(status='sampled_numpy_execution_qualified',
        exact_episode_and_trace_equality=True,approved_signatures=plan['approved_candidate_signatures'],
        qualified_new_signatures=plan['new_candidate_signatures'],games_executed=original['games']+split['games'],
        traced_games_per_trial=sum(j['n'] for j in plan['jobs'] if j['traced']),
        pure_jobs=sum(j['expected_pure_route'] for j in plan['jobs']),ordinary_jobs=sum(not j['expected_pure_route'] for j in plan['jobs']),
        trials=[{k:v for k,v in r.items() if k!='results'} for r in (original,split)],
        scope=plan['scope'],completed_at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args.source.resolve(),args.out.resolve())
    except BaseException:
        io.write(args.out/'failure.json',dict(error=traceback.format_exc(),pid=os.getpid(),at=datetime.now(timezone.utc).isoformat()))
        raise
