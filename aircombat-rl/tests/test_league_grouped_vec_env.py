from functools import partial
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from experiments.league.grouped_vec_env import GroupedSubprocVecEnv


class Tiny(gym.Env):
    render_mode = None

    def __init__(self, identity):
        self.identity = identity
        self.marker = identity
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(-100, 100, (3,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.offset = (options or {}).get('offset', 0)
        return self.obs(), dict(identity=self.identity, offset=self.offset)

    def obs(self):
        return np.array([self.identity, self.t, self.offset], np.float32)

    def step(self, action):
        self.t += 1
        done = self.t == 3 + self.identity
        return self.obs(), float(action == self.identity % 3) + .03 * self.t, done and self.identity % 2 == 0, done and self.identity % 2 == 1, dict(identity=self.identity)

    def identity_plus(self, value):
        return self.identity + value


def assert_equal(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_equal(x, y)
    else:
        assert a == b


def test_grouping_preserves_order_terminal_observations_resets_and_routed_methods():
    factories = [partial(Tiny, i) for i in range(4)]
    ordinary = SubprocVecEnv(factories, start_method='spawn')
    grouped = GroupedSubprocVecEnv(factories, processes=2)
    try:
        for vec in (ordinary, grouped):
            vec.seed(90)
            vec.set_options([dict(offset=i + 1) for i in range(4)])
        assert_equal(ordinary.reset(), grouped.reset())
        assert_equal(ordinary.reset_infos, grouped.reset_infos)
        for i in range(18):
            actions = np.array([(i + j) % 3 for j in range(4)])
            assert_equal(ordinary.step(actions), grouped.step(actions))
            assert_equal(ordinary.reset_infos, grouped.reset_infos)
        assert grouped.get_attr('identity', indices=[3, 0, 2, 3]) == [3, 0, 2, 3]
        assert grouped.env_method('identity_plus', 10, indices=[2, 0]) == [12, 10]
        grouped.set_attr('marker', 77, indices=[1, 3])
        assert grouped.get_attr('marker') == [0, 77, 2, 77]
        assert grouped.env_is_wrapped(gym.Wrapper) == [False] * 4
        assert grouped.get_attr('identity', indices=[]) == []
    finally:
        ordinary.close()
        grouped.close()


def test_same_env_count_and_rng_preserve_ppo_parameter_updates():
    torch.set_num_threads(1)
    states = []
    for grouped in (False, True):
        factories = [partial(Tiny, i) for i in range(4)]
        vec = GroupedSubprocVecEnv(factories, processes=2) if grouped else SubprocVecEnv(factories, start_method='spawn')
        try:
            model = PPO('MlpPolicy', vec, seed=811, n_steps=16, batch_size=32, n_epochs=2,
                        device='cpu', policy_kwargs=dict(net_arch=[8, 8]))
            model.learn(256)
            states.append({k: v.clone() for k, v in model.policy.state_dict().items()})
        finally:
            vec.close()
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
