"""Plan A: fixed preprocessing, finite-match training wrapper, DQN/PPO factory.

Only public observations feed the policy. JSBSim, the opponent, action space,
weapon model and official evaluation environment are unchanged.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

from aircombat_gym.wvr import obs as O
from aircombat_gym.wvr.envs.circular import CircularTargetEnv
from aircombat_gym.wvr.envs.fair import FairFightEnv


TASKS = {"fair": FairFightEnv, "circular": CircularTargetEnv}
VARIANTS = {
    "control_dqn": dict(algorithm="dqn", observation="baseline"),
    "observed_dqn": dict(algorithm="dqn", observation="relative_v1"),
    "observed_ppo": dict(algorithm="ppo", observation="relative_v1"),
}
GAMMA = 0.999
FEATURE_NAMES = (
    "range", "relative_forward", "relative_right", "bearing_sin", "bearing_cos",
    "heading_difference_sin", "heading_difference_cos",
    "range_rate", "relative_cross_speed", "own_speed", "opp_speed",
    "own_roll_sin", "own_roll_cos", "opp_roll_sin", "opp_roll_cos",
    "own_yaw_rate", "opp_yaw_rate", "own_load", "opp_load",
    "own_health", "opp_health", "own_track", "opp_track",
    "own_in_wez", "opp_in_wez", "own_boundary", "opp_boundary",
    "time_remaining", "altitude_difference", "own_vertical_speed", "opp_vertical_speed",
)


def specification(variant: str, task: str = "fair") -> dict:
    return dict(variant=variant, task=task, **VARIANTS[variant],
                gamma=GAMMA, finite_match=True, action_mode="discrete",
                feature_version=1)


class State:
    def __init__(self, spec: dict):
        self.spec = spec
        self.task = TASKS[spec["task"]]
        self.names = ("range", "own_health") if spec["observation"] == "baseline" else FEATURE_NAMES
        self.dim = len(self.names)

    def __call__(self, obs) -> np.ndarray:
        raw = np.asarray(obs, dtype=np.float32)
        if raw.shape != (O.STATE_DIM,) or not np.isfinite(raw).all():
            raise ValueError("Expected 39 finite public observation values.")
        d = O.unpack(raw)
        dx, dy, dz = (d[f"opp_{k}"] - d[f"own_{k}"] for k in ("x", "y", "h"))
        distance = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9
        if self.spec["observation"] == "baseline":
            # Exactly the original baseline transform, including its range scale.
            return np.array([math.exp(-distance / self.task.wez_r_max), d["own_health"]],
                            dtype=np.float32)
        horizontal = math.hypot(dx, dy)
        nx, ny = (dx / horizontal, dy / horizontal) if horizontal > 1e-6 else (0.0, 0.0)
        sin_yaw, cos_yaw = math.sin(d["own_psi"]), math.cos(d["own_psi"])
        dvx, dvy = d["opp_vx"] - d["own_vx"], d["opp_vy"] - d["own_vy"]
        heading_difference = d["opp_psi"] - d["own_psi"]
        speed = lambda side: math.sqrt(sum(d[f"{side}_{a}"] ** 2 for a in ("vx", "vy", "vz")))
        values = [
            distance / (distance + 10000.0),
            (dx * sin_yaw + dy * cos_yaw) / (distance + 10000.0),
            (dx * cos_yaw - dy * sin_yaw) / (distance + 10000.0),
            nx * cos_yaw - ny * sin_yaw, nx * sin_yaw + ny * cos_yaw,
            math.sin(heading_difference), math.cos(heading_difference),
            math.tanh((dvx * nx + dvy * ny) / 400.0),
            math.tanh((dvx * ny - dvy * nx) / 400.0),
            math.tanh(speed("own") / 300.0), math.tanh(speed("opp") / 300.0),
            math.sin(d["own_phi"]), math.cos(d["own_phi"]),
            math.sin(d["opp_phi"]), math.cos(d["opp_phi"]),
            math.tanh(d["own_r"] / 0.5), math.tanh(d["opp_r"] / 0.5),
            math.tanh(d["own_nz"] / 9.0), math.tanh(d["opp_nz"] / 9.0),
            d["own_health"], d["opp_health"],
            min(d["own_track_time"] / self.task.track_lock, 1.0),
            min(d["opp_track_time"] / self.task.track_lock, 1.0),
            d["own_in_wez"], d["opp_in_wez"],
            math.tanh(d["own_dist_to_boundary"] / 50000.0),
            math.tanh(d["opp_dist_to_boundary"] / 50000.0),
            d["t_remaining"] / self.task.t_max,
            math.tanh(dz / 1000.0),
            math.tanh(d["own_vz"] / 100.0), math.tanh(d["opp_vz"] / 100.0),
        ]
        # Physical bounded channels can have small numerical overshoots.
        return np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)


def potential(features) -> float:
    """Fixed A3 Phi in [-0.5, 0.5], using only the existing 31 features.

    0.25 * proximity * relative nose advantage
      + 0.125 * health advantage + 0.125 * track advantage.
    Proximity = 10000 / (range + 10000). No schedules or learned weights.
    """
    f = np.asarray(features)
    if f.shape != (31,) or not np.isfinite(f).all():
        raise ValueError("Potential requires 31 finite relative_v1 features.")
    own_cos = float(np.clip(f[4], -1.0, 1.0))
    opp_cos = float(np.clip(-(f[4] * f[6] + f[3] * f[5]), -1.0, 1.0))
    proximity = 1.0 - float(np.clip(f[0], 0.0, 1.0))
    health = float(np.clip(f[19], 0, 1) - np.clip(f[20], 0, 1))
    track = float(np.clip(f[21], 0, 1) - np.clip(f[22], 0, 1))
    return 0.25 * proximity * (own_cos - opp_cos) / 2 + 0.125 * (health + track)


class TrainingEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, spec: dict):
        super().__init__(env)
        self.training_spec = spec
        self.state = State(spec)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (self.state.dim,), np.float32)
        self.reward_mode = spec.get("reward", "terminal")
        if self.reward_mode not in ("terminal", "potential_v1"):
            raise ValueError("Unknown training reward.")
        if self.reward_mode == "potential_v1" and spec["observation"] != "relative_v1":
            raise ValueError("A3 potential requires relative_v1 observations.")
        self._previous_potential = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        features = self.state(obs)
        if self.reward_mode == "potential_v1":
            self._previous_potential = potential(features)
        return features, info

    def step(self, action):
        if self.reward_mode == "potential_v1" and self._previous_potential is None:
            raise RuntimeError("reset() is required before a shaped transition.")
        obs, _, terminated, truncated, info = self.env.step(int(action))
        won = info.get("won")
        reward = 0.0 if won is None else (1.0 if won else -0.2)
        # The 120-second match is the actual objective, not a collector cut.
        # Keep evaluation untouched, but do not bootstrap beyond match end.
        if self.training_spec["finite_match"] and truncated and info.get("outcome") == "timeout":
            terminated, truncated = True, False
        features = self.state(obs)
        if self.reward_mode == "potential_v1":
            # Actual match timeout was converted to terminated above. An external
            # collector truncation retains Phi(s') and the learner bootstrap.
            next_potential = 0.0 if terminated else potential(features)
            bonus = self.training_spec["gamma"] * next_potential - self._previous_potential
            info = dict(info, base_reward=reward, shaping_reward=bonus,
                        potential_before=self._previous_potential, potential_after=next_potential)
            reward += bonus
            self._previous_potential = None if terminated or truncated else next_potential
        return features, reward, terminated, truncated, info


def make_env(spec: dict, seed=None, shaped=True, seat="red"):
    env = TASKS[spec["task"]](action_mode="discrete", seed=seed, seat=seat)
    return TrainingEnv(env, spec) if shaped else env


def make_learner(spec: dict, seed: int, device="cpu"):
    env = Monitor(make_env(spec, seed=seed))
    common = dict(seed=seed, device=device, gamma=spec["gamma"], verbose=0,
                  policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.ReLU))
    if spec["algorithm"] == "dqn":
        return DQN("MlpPolicy", env, learning_rate=spec.get("learning_rate", 1e-3), buffer_size=50000,
                   learning_starts=1000, batch_size=32, train_freq=4, gradient_steps=1,
                   tau=1.0, target_update_interval=1000, exploration_fraction=0.1,
                   exploration_initial_eps=1.0, exploration_final_eps=0.05, **common)
    return PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=1024, batch_size=64,
               n_epochs=10, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
               target_kl=0.03, **common)


class PolicyBase:
    ACTION_MODE = "discrete"
    SPEC: dict

    def __init__(self, weights=None, device="cpu", model=None):
        self.state = State(self.SPEC)
        algorithm = DQN if self.SPEC["algorithm"] == "dqn" else PPO
        self.model = model if model is not None else algorithm.load(weights, device=device)

    def act(self, obs):
        action, _ = self.model.predict(self.state(obs), deterministic=True)
        return int(action)

    def __str__(self):
        return f"{self.SPEC['variant']} {self.state.dim}->[64,64]->9"
