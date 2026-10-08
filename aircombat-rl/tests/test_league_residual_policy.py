import numpy as np
import pytest
import torch
from gymnasium import spaces

from experiments.league.residual_features import features, numpy_logits
from experiments.league.residual_ppo_policy import ResidualPriorPolicy, actor_parameters
from experiments.plan_a.core import State, specification


def model():
    torch.set_num_threads(1)
    torch.manual_seed(91)
    return ResidualPriorPolicy(spaces.Box(-1, 1, (40,), dtype=np.float32),
                               spaces.Discrete(9), lambda _: .001)


def observations(n=72):
    rng = np.random.default_rng(43)
    x = rng.uniform(-1, 1, (n, 40)).astype(np.float32)
    x[:, -9:] = np.eye(9, dtype=np.float32)[np.arange(n) % 9]
    return torch.as_tensor(x)


def test_relative_features_preserve_existing_representation_and_action_prior():
    rng = np.random.default_rng(71)
    reference = State(specification('observed_ppo'))
    for i in range(100):
        raw = rng.normal(0, 100, 39).astype(np.float32)
        raw[30:32] = rng.uniform(0, 1, 2)
        raw[32:34] = rng.uniform(0, 5, 2)
        raw[34:36] = rng.integers(0, 2, 2)
        raw[38] = rng.uniform(0, 120)
        actual = features(raw, i % 9)
        np.testing.assert_array_equal(actual[:31], reference(raw))
        np.testing.assert_array_equal(actual[31:], np.eye(9, dtype=np.float32)[i % 9])
    with pytest.raises(ValueError): features(np.zeros(38), 0)
    with pytest.raises(ValueError): features(np.full(39, np.nan), 0)
    with pytest.raises(ValueError): features(np.zeros(39), 9)
    with pytest.raises(TypeError): features(np.zeros(39), 1.1)


def test_initial_greedy_identity_distribution_consistency_and_save_load(tmp_path):
    policy = model(); x = observations()
    with torch.no_grad():
        actions, values, log_prob = policy(x, deterministic=True)
        check_values, check_log_prob, entropy = policy.evaluate_actions(x, actions)
        probabilities = policy.get_distribution(x).distribution.probs.clone()
    assert torch.equal(actions, x[:, -9:].argmax(dim=1))
    torch.testing.assert_close(values, check_values)
    torch.testing.assert_close(log_prob, check_log_prob)
    assert torch.isfinite(entropy).all() and (probabilities > 0).all()
    assert (probabilities.max(dim=1).values > .98).all()
    path = tmp_path/'policy.pt'; policy.save(path)
    restored = ResidualPriorPolicy.load(path)
    torch.testing.assert_close(restored.get_distribution(x).distribution.probs, probabilities)
    np.testing.assert_array_equal(numpy_logits(x.numpy(), actor_parameters(policy)).argmax(axis=1), actions.numpy())


def test_residual_can_learn_actions_different_from_teacher_and_export_them(tmp_path):
    policy = model(); x = observations(); target = (x[:, -9:].argmax(dim=1)+1) % 9
    optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
    for _ in range(100):
        _, log_prob, _ = policy.evaluate_actions(x, target)
        loss = -log_prob.mean(); optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        actions, _, _ = policy(x, deterministic=True)
        latent = policy.mlp_extractor.forward_actor(policy.extract_features(x))
        logits = policy.action_net(latent)+policy.prior_bias*x[:, -9:]
    assert torch.equal(actions, target)
    parameters = actor_parameters(policy)
    path = tmp_path/'actor.npz'; np.savez_compressed(path, **parameters)
    with np.load(path, allow_pickle=False) as saved: exported = numpy_logits(x.numpy(), saved)
    np.testing.assert_allclose(exported, logits.numpy(), atol=2e-5, rtol=2e-6)
    np.testing.assert_array_equal(exported.argmax(axis=1), actions.numpy())
