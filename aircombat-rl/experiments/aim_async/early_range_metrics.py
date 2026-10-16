"""Distinct combined-condition metrics;old public-WEZ counts remain separate."""
import gymnasium as gym
from experiments.aim_async.threat_metrics import ThreatCounter
from experiments.aim_async.early_range_numpy import early_range_gate
from experiments.aim_async.features import features

METRIC_NAMES = tuple('early_range_gate_'+suffix for suffix in (
    'decisions', 'active_decisions', 'transitions', 'fraction'))


class EarlyRangeCounter(ThreatCounter):
    def record(self):
        return {key.replace('threat_gate_', 'early_range_gate_'):value
                for key, value in super().record().items()}


class EarlyRangeMetricsEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if (self.observation_space.shape != (43,) or not isinstance(self.action_space, gym.spaces.Discrete)
                or self.action_space.n != 9):
            raise ValueError('Early-range metrics require43features and9actions')
        self.counter = EarlyRangeCounter();self.next_gate = None

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.counter.reset();self.next_gate = bool(early_range_gate(state))
        return state, dict(info, **self.counter.record())

    def step(self, action):
        previous = self.next_gate
        state, reward, term, trunc, info = self.env.step(action)
        self.counter.observe(previous);self.next_gate = bool(early_range_gate(state))
        return state, reward, term, trunc, dict(info, **self.counter.record())


def early_range_gate_from_raw(raw):
    # Reuse the actual public feature builder,including float32 rounding. A
    # raw distance<=4000 comparison differs at representational boundaries.
    # The placeholder prior only affects unused last9channels;no teacher call.
    return bool(early_range_gate(features(raw,0)))
