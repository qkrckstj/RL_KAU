"""Observe public geometry and damage without changing official battle behavior."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import math,os,time,traceback
from tools import league_thread_benchmark as io
from tools.league_damage_trace import TracedEnv
from tools.league_matches import duel,initialize_worker,summary
from tools.grade import play


class GeometryEnv(TracedEnv):
    def sample_geometry(self,raw,action=None):
        dx,dy=float(raw[15]-raw[0]),float(raw[16]-raw[1]);bearing=math.atan2(dx,dy)
        wrap=lambda x:math.atan2(math.sin(x),math.cos(x))
        return dict(t=self._combat.t,range_m=math.hypot(dx,dy),own_nose_error_deg=math.degrees(wrap(bearing-float(raw[11]))),enemy_nose_error_deg=math.degrees(wrap(bearing+math.pi-float(raw[26]))),own_speed_knots=math.sqrt(sum(float(x)**2 for x in raw[3:6]))/.514444,enemy_speed_knots=math.sqrt(sum(float(x)**2 for x in raw[18:21]))/.514444,own_action=None if action is None else int(action))
    def reset(self,**kwargs):
        result=super().reset(**kwargs);self.geometry=[self.sample_geometry(result[0])];self.next_geometry=1.;return result
    def step(self,action):
        result=super().step(action)
        if self._combat.t>=self.next_geometry-1e-7 or result[2] or result[3]:
            self.geometry.append(self.sample_geometry(result[0],action));self.next_geometry=int(self._combat.t+1e-7)+1.
        return result


def observe(own,foe,band,n):
    if n<=0 or n%2:raise ValueError('Both seats required')
    episodes=[];traces=[]
    for seat in ('red','blue'):
        env=GeometryEnv(own,foe,seat)
        try:
            for seed in range(band,band+n//2):
                row=play(env,env.own_action,seed,1,seat)[0];row['seat']=seat;episodes.append(row)
                traces.append(dict(seed=seed,seat=seat,damage=env.trace.finish(),geometry_1s=env.geometry))
        finally:env.close()
    return dict(own=own['id'],foe=foe['id'],episodes=episodes,summary=summary(episodes),traces=traces)


def qualify(job):
    plain=duel(job['own'],job['foe'],job['band'],job['n']);traced=observe(**job)
    if plain['episodes']!=traced['episodes'] or plain['summary']!=traced['summary']:raise AssertionError('Instrumentation changed official results')
    return dict(job=job,exact_episode_summary_equality=True,games=2*len(plain['episodes']))


def run(source,out):
    if out.exists():raise FileExistsError(out)
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']):raise RuntimeError('Learning still live')
    previous=io.read(source/'plan.json');io.verify(previous);done=io.read(source/'completion.json');raw=io.read(source/'raw_focal_analysis.json')
    if done['status']!='focal_curriculum_pilots_complete' or raw['status']!='focal_curriculum_raw_verified':raise ValueError('Verified completed curriculum required')
    for name,digest in raw['input_sha256'].items():
        if io.sha(io.ROOT/name)!=digest:raise ValueError('Changed preceding analysis')
    baseline=previous['warm_start'];focal=[r for r in done['searches'] if r['arm']=='focal35']
    roles=dict(baseline=[r['spec'] for r in baseline['replicas']],teacher=[previous['teacher']])
    for r in focal:roles[f"focal_s{r['seed']}"]=[x['spec'] for x in r['history'][-1]['candidate']['replicas']]
    for name in ['ace','lead','pursuit','circler','fixed_left','fixed_right']:roles[name]=[next(s for s in previous['opponents'] if s['id']==name)]
    foes=[next(s for s in previous['opponents'] if s['id']==name) for name in ['temporal_extend_right','temporal_weave_right','evader']]
    band=89000000;reservation=io.ROOT/f'runs/sampled_assessment_reservation_{band}.json'
    if reservation.exists():raise FileExistsError('Diagnostic conditions reserved')
    jobs=[dict(own=s,foe=f,band=band,n=8) for specs in roles.values() for s in specs for f in foes]
    qualifier_specs=[roles['baseline'][0],roles['teacher'][0],roles[f"focal_s{focal[0]['seed']}"][0],roles[f"focal_s{focal[1]['seed']}"][0],roles['ace'][0],roles['fixed_right'][0]]
    checks=[dict(own=s,foe=f,band=170240000,n=2) for s in qualifier_specs for f in foes]
    sources=dict(previous['source_sha256']);sources[io.relative(Path(__file__).resolve())]=io.sha(Path(__file__).resolve());sources['tools/league_damage_trace.py']=io.sha(io.ROOT/'tools/league_damage_trace.py')
    inputs=dict(previous['input_sha256'])
    for p in [source/'plan.json',source/'completion.json',source/'raw_focal_analysis.json']:inputs[io.relative(p)]=io.sha(p)
    for spec in [s for specs in roles.values() for s in specs]+foes:
        if spec['kind']=='submission':
            for p in list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]:inputs[io.relative(p)]=io.sha(p)
    plan=dict(source=io.relative(source),roles=roles,opponents=foes,jobs=jobs,qualification_jobs=checks,workers=8,band=band,n=8,source_sha256=sources,input_sha256=inputs,environment=io.environment(),scope='Frozen three-foe diagnostic with fresh ICs, both seats and both action replicas. Both focal learner results included regardless of rank. Bot controls help diagnose reachable behavior. No broad/unseen-opponent/final claim or promotion. Public geometry is logged after actions, never supplied to policies.')
    io.verify(plan);io.write(reservation,dict(run=io.relative(out),band=band,stop_exclusive=band+4));io.write(out/'plan.json',plan)
    if min(io.resources()[k] for k in ['commit_headroom_gib','available_memory_gib'])<4.5:raise MemoryError('Insufficient startup headroom')
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    try:
        with ProcessPoolExecutor(max_workers=8,initializer=initialize_worker) as pool:
            futures={pool.submit(qualify,j):i for i,j in enumerate(checks)}
            for future in as_completed(futures):io.write(out/f'qualification/check_{futures[future]:03d}.json',future.result())
            io.write(out/'qualification.json',dict(status='geometry_instrumentation_matched',jobs=len(checks),games=sum(j['n']*2 for j in checks),exact_episode_summary_equality=True));io.verify(plan)
            print('geometry instrumentation qualified; diagnostic matches start',len(jobs),flush=True)
            futures={pool.submit(observe,**j):i for i,j in enumerate(jobs)}
            for count,future in enumerate(as_completed(futures),1):
                i=futures[future];io.write(out/f'matches/match_{i:03d}.json',dict(job=jobs[i],result=future.result()))
                if count%6==0:print('geometry diagnostic',count,len(jobs),flush=True)
                if io.resources()['commit_headroom_gib']<1.8:raise MemoryError('Commit safety floor')
        io.verify(plan);io.write(out/'completion.json',dict(status='weakness_geometry_complete',games=sum(j['n'] for j in jobs),qualification_games=sum(j['n']*2 for j in checks),policy_promoted=False,final_opened=False,heldout_opened=False))
    except BaseException:io.write(out/'failure.json',dict(traceback=traceback.format_exc(),resources=io.resources()));raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source.resolve(),a.out.resolve())
