import torch
from experiments.plan_a.core import make_learner, specification
from tools.plan_a_stability import protocol


def test_learning_rate_changes_optimizer_only():
    torch.set_num_threads(1)
    spec = specification("observed_dqn")
    a = make_learner(spec, 17)
    b = make_learner(dict(spec, learning_rate=1e-4), 17)
    try:
        assert a.policy.optimizer.param_groups[0]["lr"] == 1e-3
        assert b.policy.optimizer.param_groups[0]["lr"] == 1e-4
        assert all(torch.equal(v, b.policy.state_dict()[k]) for k, v in a.policy.state_dict().items())
        for attr in ("gamma", "batch_size", "buffer_size", "train_freq", "target_update_interval"):
            assert getattr(a, attr) == getattr(b, attr)
    finally:
        a.get_env().close()
        b.get_env().close()


def test_smoke_does_not_open_heldout_test():
    assert protocol(True)["test_band"] == 900000
    assert protocol(False)["test_band"] == 1400000
