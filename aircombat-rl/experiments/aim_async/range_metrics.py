"""Distinct combined-condition metrics;old public-WEZ counts remain separate."""
import gymnasium as gym
from experiments.aim_async.threat_metrics import ThreatCounter
from experiments.aim_async.range_numpy import range_gate,range_limit
from experiments.aim_async.features import features

METRIC_NAMES = tuple('range_gate_'+suffix for suffix in (
    'decisions', 'active_decisions', 'transitions', 'fraction'))


class RangeCounter(ThreatCounter):
    def record(self):
        return {key.replace('threat_gate_', 'range_gate_'):value
                for key, value in super().record().items()}


class RangeMetricsEnv(gym.Wrapper):
    def __init__(self, env, *, range_threshold_m):
        range_limit(range_threshold_m)
        self.range_threshold_m=float(range_threshold_m)
        super().__init__(env)
        if (self.observation_space.shape != (43,) or not isinstance(self.action_space, gym.spaces.Discrete)
                or self.action_space.n != 9):
            raise ValueError('Early-range metrics require43features and9actions')
        self.counter = RangeCounter();self.next_gate = None

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.counter.reset();self.next_gate = bool(range_gate(state,self.range_threshold_m))
        return state, dict(info, **self.counter.record())

    def step(self, action):
        previous = self.next_gate
        state, reward, term, trunc, info = self.env.step(action)
        self.counter.observe(previous);self.next_gate = bool(range_gate(state,self.range_threshold_m))
        return state, reward, term, trunc, dict(info, **self.counter.record())


def range_gate_from_raw(raw,range_threshold_m):
    # Reuse the actual public feature builder,including float32 rounding. A
    # raw distance<=threshold comparison differs at representational boundaries.
    # The placeholder prior only affects unused last9channels;no teacher call.
    return bool(range_gate(features(raw,0),range_threshold_m))
