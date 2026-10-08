"""Several independent Gymnasium environments per SB3 subprocess.

Grouping reduces process count and per-transition RPC/action-forward overhead.
Each inner environment retains SB3 DummyVecEnv's terminal/reset handling.
This adapter deliberately supports the project's Box observations and Discrete
actions only. Changing the number of environments changes rollout composition;
it is not an assertion that training trajectories or policy quality stay equal.
"""
from functools import partial
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


class _Group(gym.Env):
    metadata = {'render_modes': []}
    render_mode = None

    def __init__(self, factories):
        self.vec = DummyVecEnv(factories)
        # Project rewards are Python floats. SubprocVecEnv stacks them as
        # float64; avoid DummyVecEnv's earlier float32 rounding here.
        self.vec.buf_rews = np.zeros(len(factories), dtype=np.float64)
        obs, action = self.vec.observation_space, self.vec.action_space
        if not isinstance(obs, gym.spaces.Box) or not isinstance(action, gym.spaces.Discrete):
            self.vec.close()
            raise TypeError('Grouped league environments require Box observations and Discrete actions')
        self.observation_space = gym.spaces.Box(
            np.stack([obs.low] * len(factories)), np.stack([obs.high] * len(factories)), dtype=obs.dtype)
        self.action_space = gym.spaces.MultiDiscrete([action.n] * len(factories))

    def reset(self, *, seed=None, options=None):
        options = options or {}
        self.vec._seeds = list(options.get('inner_seeds', [None] * self.vec.num_envs))
        self.vec._options = list(options.get('inner_options', [{} for _ in range(self.vec.num_envs)]))
        observations = self.vec.reset()
        return observations, {'inner_reset_infos': self.vec.reset_infos}

    def step(self, actions):
        observations, rewards, dones, infos = self.vec.step(actions)
        # The outer transport environment never terminates. All actual episode
        # endings and terminal observations remain inside the per-env records.
        return observations, 0., False, False, dict(
            inner_rewards=rewards, inner_dones=dones, inner_infos=infos,
            inner_reset_infos=self.vec.reset_infos)

    def inner_call(self, command, indices, args, kwargs):
        return getattr(self.vec, command)(*args, indices=indices, **kwargs)

    def close(self):
        self.vec.close()


class GroupedSubprocVecEnv(VecEnv):
    """N environments on P processes, with stable contiguous environment order."""
    def __init__(self, env_fns, processes=4):
        if not env_fns or not 1 <= processes <= len(env_fns) or len(env_fns) % processes:
            raise ValueError('A positive environment count divisible by process count is required')
        self.group_size = len(env_fns) // processes
        self.process_count = processes
        self.closed = False
        self.waiting = False
        self.transport = SubprocVecEnv([
            partial(_Group, env_fns[i:i + self.group_size])
            for i in range(0, len(env_fns), self.group_size)], start_method='spawn')
        obs = self.transport.observation_space
        observation_space = gym.spaces.Box(obs.low[0], obs.high[0], dtype=obs.dtype)
        action_space = gym.spaces.Discrete(int(self.transport.action_space.nvec[0]))
        super().__init__(len(env_fns), observation_space, action_space)

    def reset(self):
        self.transport.set_options([
            dict(inner_seeds=self._seeds[i:i + self.group_size],
                 inner_options=self._options[i:i + self.group_size])
            for i in range(0, self.num_envs, self.group_size)])
        observations = self.transport.reset()
        self.reset_infos = [info for group in self.transport.reset_infos for info in group['inner_reset_infos']]
        self._reset_seeds()
        self._reset_options()
        return observations.reshape((self.num_envs,) + self.observation_space.shape)

    def step_async(self, actions):
        if self.waiting:
            raise RuntimeError('A grouped environment step is already pending')
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,):
            raise ValueError('One discrete action per environment is required')
        self.transport.step_async(actions.reshape(self.process_count, self.group_size))
        self.waiting = True

    def step_wait(self):
        if not self.waiting:
            raise RuntimeError('No grouped environment step is pending')
        observations, _, _, groups = self.transport.step_wait()
        self.waiting = False
        self.reset_infos = [info for group in groups for info in group['inner_reset_infos']]
        return (observations.reshape((self.num_envs,) + self.observation_space.shape),
                np.concatenate([group['inner_rewards'] for group in groups]),
                np.concatenate([group['inner_dones'] for group in groups]),
                tuple(info for group in groups for info in group['inner_infos']))

    def _route(self, command, indices, args, kwargs):
        selected = list(self._get_indices(indices))
        if any(not 0 <= i < self.num_envs for i in selected):
            raise IndexError('Environment index out of range')
        answer = [None] * len(selected)
        groups = {}
        for position, index in enumerate(selected):
            groups.setdefault(index // self.group_size, []).append((position, index % self.group_size))
        for group, positions in groups.items():
            result = self.transport.env_method('inner_call', command,
                [local for _, local in positions], args, kwargs, indices=[group])[0]
            if result is not None:
                for (position, _), value in zip(positions, result, strict=True):
                    answer[position] = value
        return answer

    def get_attr(self, attr_name, indices=None):
        return self._route('get_attr', indices, (attr_name,), {})

    def set_attr(self, attr_name, value, indices=None):
        self._route('set_attr', indices, (attr_name, value), {})

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self._route('env_method', indices, (method_name, *method_args), method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self._route('env_is_wrapped', indices, (wrapper_class,), {})

    def get_images(self):
        return self.env_method('render')

    def close(self):
        if self.closed:
            return
        self.transport.close()
        self.waiting = False
        self.closed = True
