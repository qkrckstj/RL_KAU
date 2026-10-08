import gymnasium as gym
import pytest
from experiments.league.action_hold_worker import HoldDecisions


class Physical(gym.Env):
    observation_space=gym.spaces.Discrete(20)
    action_space=gym.spaces.Discrete(9)
    def __init__(self,end=100):self.end=end;self.calls=[]
    def reset(self,**kwargs):self.t=0;return 0,{}
    def step(self,action):
        assert self.t<self.end
        self.t+=1;self.calls.append(action)
        return self.t,float(self.t),self.t==self.end,False,{'official_tick':self.t}


def test_discount_and_every_official_step_preserved():
    raw=Physical();wrapped=HoldDecisions(raw,5,.9);wrapped.reset()
    obs,reward,terminated,truncated,info=wrapped.step(6)
    assert raw.calls==[6]*5 and obs==5 and not terminated and not truncated
    assert reward==pytest.approx(sum(.9**i*(i+1) for i in range(5)))
    assert info['physical_steps']==5 and wrapped.physical_status()['decision_steps']==1


def test_terminal_break_and_reset_no_leak():
    raw=Physical(end=3);wrapped=HoldDecisions(raw,5,.9);wrapped.reset()
    obs,reward,terminated,truncated,info=wrapped.step(2)
    assert terminated and not truncated and obs==3 and raw.calls==[2]*3
    assert reward==pytest.approx(1+.9*2+.9**2*3) and info['physical_steps']==3
    wrapped.reset();assert wrapped.physical_status()['episode_physical_steps']==0
    assert wrapped.physical_status()['physical_steps']==3


def test_single_step_has_original_transition():
    a=Physical(end=3);b=Physical(end=3);wrapped=HoldDecisions(a,1,.999)
    wrapped.reset();b.reset()
    for action in [8,0,4]:
        x=wrapped.step(action);y=b.step(action)
        assert x[:4]==y[:4] and x[4]['official_tick']==y[4]['official_tick']
