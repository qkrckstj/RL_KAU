"""A3 invariants: reward telescoping, reset/terminal handling and unchanged inputs."""
import math

import gymnasium as gym
import numpy as np
import pytest
import torch

from aircombat_gym.wvr import obs as O
from experiments.plan_a.core import (State, TrainingEnv, make_env, make_learner,
                                      potential, specification)
from tools.plan_a_shaping import protocol, freeze
from tools.plan_a import write_json


def config():
    return dict(specification("observed_dqn"), reward="potential_v1")


def raw_state(x=1000, own_heading=math.pi / 2, opp_heading=math.pi / 2):
    raw = np.zeros(39, dtype=np.float32)
    for name, value in dict(opp_x=x, own_psi=own_heading, opp_psi=opp_heading,
                            own_health=1, opp_health=0.7, own_track_time=0.3,
                            t_remaining=100).items():
        raw[O.index(name)] = value
    return raw


class Sequence(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (39,), np.float32)
    action_space = gym.spaces.Discrete(9)

    def __init__(self, end="kill"):
        self.end = end
        self.initial = raw_state()

    def reset(self, **kwargs):
        self.i = 0
        return self.initial.copy(), {}

    def step(self, action):
        self.i += 1
        done = self.i == 3
        info = {} if not done or self.end == "external" else {
            "outcome": self.end, "won": self.end == "kill"}
        return raw_state(x=1000 - self.i * 200), 0.0, (
            done and self.end in ("kill", "died", "mutual")), (
            done and self.end in ("timeout", "external")), info


@pytest.mark.parametrize("end", ["kill", "died", "mutual", "timeout"])
def test_discounted_shaping_telescopes_from_first_step_to_terminal(end):
    spec = config()
    spec["gamma"] = 0.93  # Deliberately nondefault: wrapper must use learner spec.
    env = TrainingEnv(Sequence(end), spec)
    obs, _ = env.reset()
    initial = potential(obs)
    discounted_bonus = 0.0
    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(0)
        discounted_bonus += spec["gamma"] ** i * info["shaping_reward"]
        assert reward == pytest.approx(info["base_reward"] + info["shaping_reward"])
        if i == 0:
            assert info["potential_before"] == initial
    assert terminated and not truncated
    assert info["potential_after"] == 0
    assert discounted_bonus == pytest.approx(-initial, abs=1e-12)
    assert info["base_reward"] == (1.0 if end == "kill" else -0.2)


def test_external_cut_retains_bootstrap_boundary_term():
    spec = config()
    env = TrainingEnv(Sequence("external"), spec)
    obs, _ = env.reset()
    initial = potential(obs)
    bonus = 0
    for i in range(3):
        obs, _, term, trunc, info = env.step(0)
        bonus += spec["gamma"] ** i * info["shaping_reward"]
    assert not term and trunc
    assert info["potential_after"] == potential(obs) != 0
    assert bonus == pytest.approx(-initial + spec["gamma"] ** 3 * potential(obs))


def test_reset_replaces_potential_and_done_requires_reset():
    base = Sequence()
    env = TrainingEnv(base, config())
    env.reset()
    for _ in range(3):
        env.step(0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    base.initial = raw_state(9000, own_heading=0)
    obs, _ = env.reset()
    _, _, _, _, info = env.step(0)
    assert info["potential_before"] == potential(obs)


def test_potential_is_bounded_and_antisymmetric_between_seats():
    state = State(config())
    rng = np.random.default_rng(314)
    for _ in range(100):
        raw = raw_state(rng.uniform(0, 50000), *rng.uniform(-math.pi, math.pi, 2))
        for side in ("own", "opp"):
            raw[O.index(f"{side}_health")] = rng.uniform()
            raw[O.index(f"{side}_track_time")] = rng.uniform(0, 5)
        phi = potential(state(raw))
        assert abs(phi) <= 0.5
        assert potential(state(O.mirror(raw))) == pytest.approx(-phi, abs=2e-7)
    coincident = raw_state(0)
    assert math.isfinite(potential(state(coincident)))


def test_chasing_orientation_scores_above_being_chased():
    state = State(config())
    favorable = raw_state()
    unfavorable = raw_state(own_heading=-math.pi / 2, opp_heading=-math.pi / 2)
    assert potential(state(favorable)) > potential(state(unfavorable))


@pytest.mark.parametrize("seat", ["red", "blue"])
def test_reward_change_does_not_change_observations_dynamics_or_actions(seat):
    plain = make_env(specification("observed_dqn"), seed=123, seat=seat)
    shaped = make_env(config(), seed=123, seat=seat)
    try:
        a, _ = plain.reset(seed=123)
        b, _ = shaped.reset(seed=123)
        np.testing.assert_array_equal(a, b)
        for i in range(64):
            a, r, t, u, info = plain.step(i % 9)
            b, s, v, w, extra = shaped.step(i % 9)
            np.testing.assert_array_equal(a, b)
            assert (t, u) == (v, w)
            assert s - r == pytest.approx(extra["shaping_reward"])
            assert {k: extra[k] for k in info} == info
        assert plain.action_space == shaped.action_space
    finally:
        plain.close()
        shaped.close()


def test_shaping_keeps_initial_network_and_hyperparameters_identical():
    torch.set_num_threads(1)
    a = make_learner(specification("observed_dqn"), 7)
    b = make_learner(config(), 7)
    try:
        for key, value in a.policy.state_dict().items():
            assert torch.equal(value, b.policy.state_dict()[key])
        for name in ("gamma", "learning_rate", "buffer_size", "batch_size",
                     "train_freq", "gradient_steps", "target_update_interval",
                     "learning_starts", "exploration_fraction", "exploration_final_eps"):
            assert getattr(a, name) == getattr(b, name)
        assert b.env.envs[0].env.training_spec["gamma"] == b.gamma
    finally:
        a.get_env().close()
        b.get_env().close()


def test_unknown_reward_and_wrong_observation_are_rejected():
    with pytest.raises(ValueError):
        TrainingEnv(Sequence(), dict(config(), reward="typo"))
    with pytest.raises(ValueError):
        TrainingEnv(Sequence(), dict(config(), observation="baseline"))


def test_smoke_never_opens_reserved_test_band():
    assert protocol(True)["test_band"] == 900000
    assert protocol(False)["test_band"] == 1300000


def test_selection_freezes_validation_choices_and_detects_replaced_weights(tmp_path):
    plan = dict(seeds=[0], budgets=[4, 8], test_band=1300000, test_n=40)
    for reward in ("terminal", "potential_v1"):
        dest = tmp_path / reward / "s0"
        write_json(dest / "result.json", dict(status="complete", steps=8))
        # Untrained and later-best scores must not leak into budget 4.
        history = [dict(step=0, summary=dict(kills=20, t_kill=1)),
                   dict(step=4, summary=dict(kills=2, t_kill=50)),
                   dict(step=8, summary=dict(kills=4, t_kill=40))]
        write_json(dest / "validation.json", history)
        for step in (4, 8):
            p = dest / f"checkpoints/step_{step}/policy_net.zip"
            p.parent.mkdir(parents=True)
            p.write_bytes(f"model-{step}".encode())
        for name in ("policy.py", "wrappers.py", "utils.py", "config.json"):
            (dest / name).write_text("fixture", encoding="utf-8")
    frozen = freeze(tmp_path, plan)
    assert frozen["selected_reward"] == "terminal"  # Prespecified tie.
    assert all(c["step"] <= c["budget"] for c in frozen["choices"])
    assert freeze(tmp_path, plan) == frozen
    (tmp_path / frozen["selected_policy"]["weights"]).write_bytes(b"replaced")
    with pytest.raises(ValueError, match="changed"):
        freeze(tmp_path, plan)
