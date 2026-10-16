"""Distinct combined-condition metrics;old public-WEZ counts remain separate."""
import gymnasium as gym
from experiments.aim_async.threat_metrics import ThreatCounter
from experiments.aim_async.wez_or_damaged_numpy import wez_or_damaged_gate

METRIC_NAMES = tuple('wez_or_damaged_gate_'+suffix for suffix in (
    'decisions', 'active_decisions', 'transitions', 'fraction'))


class WezOrDamagedCounter(ThreatCounter):
    def record(self):
        return {key.replace('threat_gate_', 'wez_or_damaged_gate_'):value
                for key, value in super().record().items()}


class WezOrDamagedMetricsEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if (self.observation_space.shape != (43,) or not isinstance(self.action_space, gym.spaces.Discrete)
                or self.action_space.n != 9):
            raise ValueError('Combined metrics require43features and9actions')
        self.counter = WezOrDamagedCounter();self.next_gate = None

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.counter.reset();self.next_gate = bool(wez_or_damaged_gate(state))
        return state, dict(info, **self.counter.record())

    def step(self, action):
        previous = self.next_gate
        state, reward, term, trunc, info = self.env.step(action)
        self.counter.observe(previous);self.next_gate = bool(wez_or_damaged_gate(state))
        return state, reward, term, trunc, dict(info, **self.counter.record())
