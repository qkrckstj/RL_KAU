"""Compare complete real flights under repeated identical official commands."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from experiments.league.action_hold_worker import make_hold_worker
from experiments.league.strategy_worker import make_strategy_worker
from tools import league_thread_benchmark as io


def run(source):
    old=io.read(source/'plan.json');io.verify(old);checks=[]
    for hold in [1,5]:
        for seat in range(2):
            plan=dict(old,seed=6799,ic_band=170340000,workers=2,hold_steps=hold,opponents=[next(s for s in old['opponents'] if s['id']=='ace')],probabilities=[1.],ppo=dict(old['ppo'],gamma=old['physical_gamma']))
            direct=make_strategy_worker(plan,seat,False);repeated=make_hold_worker(plan,seat,False)
            try:
                a,_=direct.reset();b,_=repeated.reset();assert np.array_equal(a,b)
                done=False;decisions=0;physical=0
                while not done:
                    action=[0,4,8,2,6][decisions%5];reward=0.;used=0
                    for i in range(hold):
                        a,r,terminated,truncated,info=direct.step(action);reward+=plan['physical_gamma']**i*r;used+=1
                        if terminated or truncated:break
                    b,total,t,u,other=repeated.step(action)
                    assert np.array_equal(a,b) and total==reward and (t,u)==(terminated,truncated)
                    for key in ['outcome','won','own_health','opp_health','t','official_reward','official_episode_flags']:
                        assert info.get(key)==other.get(key),(hold,seat,key)
                    assert other['physical_steps']==used
                    physical+=used;decisions+=1;done=t or u
                checks.append(dict(hold_steps=hold,seat=seat,physical_steps=physical,decisions=decisions,outcome=other['outcome'],observation_reward_clock_health_verdict_exact=True))
            finally:direct.close();repeated.close()
    result=dict(status='real_hold_physics_matched',game_executions=8,checks=checks,source_sha256=io.sha(Path(__file__).resolve()),plan_sha256=io.sha(source/'plan.json'))
    target=source/'real_physics_equivalence.json'
    if target.exists():raise FileExistsError(target)
    io.write(target,result);print(result)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('source',type=Path);a=p.parse_args();run(a.source.resolve())
