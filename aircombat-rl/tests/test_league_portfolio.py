"""Causal public-motion gate, expert state, and episode reset checks."""
import numpy as np
from experiments.league.portfolio_policy import Policy


class Expert:
    def __init__(self,action):self.action=action;self.calls=0
    def reset(self):self.calls=0
    def act(self,obs):self.calls+=1;return self.action


def policy(gate):
    p=Policy.__new__(Policy);p.experts=[Expert(i) for i in range(4)];p.probe=.5;p.gate=np.array(gate);p.forced=None;p.reset();return p


def observation(t,yaw=.0):
    x=np.zeros(39,np.float32);x[16]=10000-200*t;x[19]=200;x[26]=yaw*t;x[29]=yaw;x[38]=120-t;return x


def test_stateful_prefix_and_episode_reset():
    p=policy([[1,0,0,0,0],[-2,0,0,0,0],[-2,0,0,0,0]])
    assert p.act(observation(0))==0
    assert p.act(observation(.25))==0
    assert p.act(observation(.5))==1 and p.selected==1
    assert [e.calls for e in p.experts]==[2,3,2,2]
    assert p.act(observation(1))==1
    assert p.act(observation(0))==0 and p.selected is None
    assert [e.calls for e in p.experts]==[1,1,1,1]


def test_direction_invariant_features_and_constant_base():
    gates=[[-8,0,0,0,0]]*3
    left,right=policy(gates),policy(gates)
    for t in [0,.25,.5,1.]:
        assert left.act(observation(t,.2))==right.act(observation(t,-.2))==0
    np.testing.assert_allclose(left.features,right.features)
    assert left.selected==right.selected==0
