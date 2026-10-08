"""Holding a legal action must preserve observation updates and clear on reset."""
from tools.league_action_hold_probe import SUFFIX


class CountingPolicy:
    def reset(self):self.previous=None;self.calls=0
    def act(self,obs):
        if self.previous is not None and obs[38]>self.previous:self.reset()
        self.previous=obs[38];action=self.calls%9;self.calls+=1;return action


def held(steps):
    namespace={'Policy':CountingPolicy};exec(SUFFIX,namespace)
    policy=namespace['Policy'].__new__(namespace['Policy']);policy.hold_steps=steps;policy.reset();return policy


def obs(step):return [0.]*38+[120.-.05*step]


def test_every_observation_updates_but_action_is_held():
    p=held(5)
    assert [p.act(obs(i)) for i in range(11)]==[0]*5+[5]*5+[1]
    assert p.calls==11
    assert p.act(obs(0))==0 and p.calls==1
    assert p.act(obs(1))==0


def test_one_step_is_original_and_explicit_reset_clears_hold():
    p=held(1);direct=CountingPolicy();direct.reset()
    for i in range(20):assert p.act(obs(i))==direct.act(obs(i))
    p.reset();assert p.act(obs(0))==0
