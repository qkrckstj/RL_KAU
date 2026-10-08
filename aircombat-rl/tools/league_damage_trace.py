"""Observe damage timing without changing policy observations, actions or verdicts."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import shutil
import time
import traceback
from tools.autolab_cem import ROOT, read, write, sha
from tools.league_matches import MatchEnv, duel, initialize_worker, summary
from tools.grade import play
from tools.league_train import verify, unique_entrants
from tools.league_validate import process_live


class DamageTrace:
    def __init__(self, t, hp, enemy_hp):
        self.first_hit = None
        self.first_enemy_hit = None
        self.minimum_hp = hp
        self.previous = (t,hp,enemy_hp)
        self.damage_seconds = 0.
        self.snapshots = {str(t):None for t in (10,20,30,60)}
        self.timeline = [dict(t=t,hp=hp,enemy_hp=enemy_hp)]
        self.next_sample = 1.

    def observe(self,t,hp,enemy_hp):
        old_t,old_hp,old_enemy = self.previous
        if t < old_t: raise ValueError('Nonmonotonic episode clock')
        if hp < old_hp-1e-10:
            if self.first_hit is None: self.first_hit=t
            self.damage_seconds += t-old_t
        if enemy_hp < old_enemy-1e-10 and self.first_enemy_hit is None:
            self.first_enemy_hit=t
        self.minimum_hp=min(self.minimum_hp,hp)
        for milestone in self.snapshots:
            if self.snapshots[milestone] is None and t>=float(milestone)-1e-7:
                self.snapshots[milestone]=dict(t=t,hp=hp,enemy_hp=enemy_hp)
        if t>=self.next_sample-1e-7:
            self.timeline.append(dict(t=t,hp=hp,enemy_hp=enemy_hp))
            self.next_sample=int(t+1e-7)+1.
        self.previous=(t,hp,enemy_hp)

    def finish(self):
        t,hp,enemy_hp=self.previous
        return dict(first_hit=self.first_hit,first_enemy_hit=self.first_enemy_hit,
            minimum_hp=self.minimum_hp,damage_seconds=self.damage_seconds,
            snapshots=self.snapshots,timeline_1s=self.timeline,
            terminal=dict(t=t,hp=hp,enemy_hp=enemy_hp),
            snapshot_rule='Null if the engagement ended before that time; never extrapolate terminal HP to later times.')


class TracedEnv(MatchEnv):
    def reset(self,**kwargs):
        result=super().reset(**kwargs)
        self.trace=DamageTrace(self._combat.t,self._combat.health[self.seat],
                              self._combat.health['blue' if self.seat=='red' else 'red'])
        return result

    def step(self,action):
        result=super().step(action)
        self.trace.observe(self._combat.t,self._combat.health[self.seat],
                           self._combat.health['blue' if self.seat=='red' else 'red'])
        return result


def observe(own,foe,band,n):
    if n<=0 or n%2: raise ValueError('Complete seat pairs required')
    episodes=[];traces=[]
    for seat in ('red','blue'):
        env=TracedEnv(own,foe,seat)
        try:
            for seed in range(band,band+n//2):
                row=play(env,env.own_action,seed,1,seat)[0]
                row['seat']=seat;episodes.append(row)
                traces.append(dict(seed=seed,seat=seat,**env.trace.finish()))
        finally:env.close()
    return dict(own=own['id'],foe=foe['id'],episodes=episodes,
                summary=summary(episodes),damage_traces=traces)


def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def run(search,out):
    if (out/'runtime.json').exists() and process_live(read(out/'runtime.json')['pid']):
        raise ValueError('Damage diagnostic already running')
    if (out/'completion.json').exists():return
    digest=sha(Path(__file__))
    write(out/'runtime.json',dict(pid=os.getpid(),workers_when_active=3))
    while True:
        live=[]
        for folder in (search,search/'novel_audit'):
            if (folder/'runtime.json').exists():
                pid=read(folder/'runtime.json')['pid']
                if process_live(pid):live.append(pid)
        if not live:break
        write(out/'progress.json',dict(stage='waiting_for_prior_controllers',pids=live))
        time.sleep(45)
        if sha(Path(__file__))!=digest:raise ValueError('Queued diagnostic source changed')
    for folder in (search,search/'novel_audit'):
        if not (folder/'completion.json').exists():raise ValueError('Predecessor did not complete')
    previous=read(search/'plan.json');verify(previous)
    chosen=read(search/'frozen_selection.json')['chosen']
    original=next(x for x in previous['opponents'] if x['id']=='cem_original')
    models=unique_entrants([original,previous['parent'],previous['reference'],chosen])
    foes=[x for x in previous['opponents'] if x['id'] in ('ace','lead','ddqn_s1','refine_2201','evader')]
    sources=dict(previous['source_sha256']);sources[relative(Path(__file__))]=digest
    inputs=dict(previous['input_sha256'])
    for p in (search/'plan.json',search/'completion.json',search/'frozen_selection.json'):
        inputs[relative(p)]=sha(p)
    for model in models:
        if model['kind']=='submission':
            folder=ROOT/model['design']
            for p in list(folder.glob('*.py'))+[ROOT/model['weights']]:inputs[relative(p)]=sha(p)
    plan=dict(models=models,opponents=foes,band=38000000,n=40,workers=3,
        compatibility_band=37990000,source_sha256=sources,input_sha256=inputs,
        scope='Open-development damage timing. Not model selection or an unseen performance test. Actual post-step health is observed only by the logger; original policy inputs and verdicts are returned untouched.')
    if (out/'plan.json').exists() and read(out/'plan.json')!=plan:raise ValueError('Changed diagnostic plan')
    write(out/'plan.json',plan)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    if not (out/'compatibility.json').exists():
        write(out/'progress.json',dict(stage='verifying_no_behavior_change'))
        ace=next(x for x in foes if x['id']=='ace')
        tests=[]
        for model in (previous['parent'],chosen):
            native=duel(model,ace,plan['compatibility_band'],2)
            traced=observe(model,ace,plan['compatibility_band'],2)
            if native['episodes']!=traced['episodes']:raise ValueError('Logger changed official episode records')
            tests.append(dict(model=model['id'],native=native,traced=traced))
        write(out/'compatibility.json',dict(status='passed',checks=tests))
    results={}
    with ProcessPoolExecutor(max_workers=3,initializer=initialize_worker) as pool:
        pending={}
        jobs=[(m,f) for m in models for f in foes]
        for i,(model,foe) in enumerate(jobs):
            path=out/f'match_{i:03d}.json'
            if path.exists():results[i]=read(path)
            else:pending[pool.submit(observe,model,foe,plan['band'],plan['n'])]=(i,path)
        for future in as_completed(pending):
            i,path=pending[future];results[i]=future.result();write(path,results[i])
            write(out/'progress.json',dict(stage='damage_timing',done=len(results),total=len(jobs)))
    verify(plan)
    write(out/'completion.json',dict(status='diagnostic_complete',matches=len(results),games=sum(x['summary']['n'] for x in results.values()),scope=plan['scope'],next='Analyze damage timing jointly with paired WDL; preserve early-ended episodes as censored snapshots. No automatic promotion.'))
    write(out/'progress.json',dict(stage='complete'))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--search',type=Path,default=Path('runs/league_interception_train_20261006'))
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    try:run(args.search.resolve(),args.out.resolve())
    except Exception:
        write(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
