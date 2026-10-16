"""Non-intervening gate metrics on the state that precedes each action."""
import gymnasium as gym
from experiments.aim_async.threat_prior_numpy import threat_gate

METRIC_NAMES=('threat_gate_decisions','threat_gate_active_decisions',
              'threat_gate_transitions','threat_gate_fraction')


class ThreatCounter:
    def __init__(self):self.reset()

    def reset(self):
        self.decisions=0;self.active=0;self.transitions=0;self.previous=None

    def observe(self,gate):
        gate=bool(gate)
        self.decisions+=1;self.active+=int(gate)
        self.transitions+=int(self.previous is not None and gate!=self.previous)
        self.previous=gate

    def record(self):
        return dict(threat_gate_decisions=self.decisions,threat_gate_active_decisions=self.active,
                    threat_gate_transitions=self.transitions,
                    threat_gate_fraction=self.active/self.decisions if self.decisions else 0.)


class ThreatMetricsEnv(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        if (self.observation_space.shape!=(43,) or not isinstance(self.action_space,gym.spaces.Discrete)
                or self.action_space.n!=9):
            raise ValueError('Threat metrics require43features and9actions')
        self.counter=ThreatCounter();self.next_gate=None

    def reset(self,**kwargs):
        state,info=self.env.reset(**kwargs)
        self.counter.reset();self.next_gate=bool(threat_gate(state))
        return state,dict(info,**self.counter.record())

    def step(self,action):
        previous=self.next_gate
        state,reward,term,trunc,info=self.env.step(action)
        self.counter.observe(previous);self.next_gate=bool(threat_gate(state))
        return state,reward,term,trunc,dict(info,**self.counter.record())
