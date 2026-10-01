"""Guard preprocessing invariants and the finite-match learning contract."""
import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from aircombat_gym.wvr import obs as O
from experiments.plan_a.core import State, TrainingEnv, make_env, specification


def test_control_observation_matches_original_baseline():
    path = Path(__file__).resolve().parents[1] / "templates/project_04_fair/baseline/wrappers.py"
    module_spec = importlib.util.spec_from_file_location("original_fair_wrappers", path)
    original = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(original)
    config = specification("control_dqn")
    env = make_env(config, shaped=False)
    try:
        raw, _ = env.reset(seed=123)
        transform = State(config)
        for i in range(30):
            np.testing.assert_array_equal(transform(raw), original.State()(raw))
            raw, _, _, _, _ = env.step(i % 9)
    finally:
        env.close()


def test_relative_features_ignore_global_rotation_and_translation():
    config = specification("observed_dqn")
    env = make_env(config, shaped=False)
    try:
        raw, _ = env.reset(seed=456)
        changed = raw.copy()
        angle = 1.2
        c, s = np.cos(angle), np.sin(angle)
        for side in ("own", "opp"):
            for a, b in (("x", "y"), ("vx", "vy")):
                ia, ib = O.index(f"{side}_{a}"), O.index(f"{side}_{b}")
                changed[ia] = c * raw[ia] + s * raw[ib]
                changed[ib] = -s * raw[ia] + c * raw[ib]
            changed[O.index(f"{side}_psi")] += angle
            changed[O.index(f"{side}_x")] += 1234.0
            changed[O.index(f"{side}_y")] -= 2345.0
        # Supplied arena-distance channels stay unchanged; this tests geometry,
        # not a physical translation to a different part of the arena.
        np.testing.assert_allclose(State(config)(raw), State(config)(changed), atol=2e-6)
    finally:
        env.close()


@pytest.mark.parametrize("seat", ["red", "blue"])
def test_features_are_finite_bounded_and_identical_between_train_and_policy(seat):
    config = specification("observed_dqn")
    env = make_env(config, seed=789, shaped=False, seat=seat)
    wrapped = TrainingEnv(env, config)
    try:
        features, _ = wrapped.reset(seed=789)
        assert features.shape == (31,)
        assert wrapped.observation_space.contains(features)
        for i in range(100):
            features, _, term, trunc, info = wrapped.step(i % 9)
            assert wrapped.observation_space.contains(features)
            raw = env.obs_for(seat, env._last)
            np.testing.assert_array_equal(features, State(config)(raw))
            if term or trunc:
                wrapped.reset(seed=790 + i)
    finally:
        wrapped.close()


class EndEvent(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (39,), np.float32)
    action_space = gym.spaces.Discrete(9)

    def __init__(self, outcome):
        self.outcome = outcome

    def step(self, action):
        info = {} if self.outcome is None else dict(outcome=self.outcome, won=False)
        return np.zeros(39, dtype=np.float32), 0.0, False, True, info


def test_only_actual_match_timeout_becomes_terminal():
    config = specification("observed_dqn")
    _, reward, term, trunc, _ = TrainingEnv(EndEvent("timeout"), config).step(0)
    assert (reward, term, trunc) == (-0.2, True, False)
    _, reward, term, trunc, _ = TrainingEnv(EndEvent(None), config).step(0)
    assert (reward, term, trunc) == (0.0, False, True)


def test_observation_rejects_nonfinite_input():
    raw = np.zeros(39, dtype=np.float32)
    raw[0] = np.nan
    with pytest.raises(ValueError):
        State(specification("observed_ppo"))(raw)
