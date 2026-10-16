"""Override effective PPO update rate without rebuilding initialized Adam state."""
import math
from experiments.aim_async.reference_ppo import ReferenceAnchoredPPO


def load_with_rate(path, *, env, seed, learning_rate):
    rate = float(learning_rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('Learning rate must be finite and positive')
    model = ReferenceAnchoredPPO.load(
        path, env=env, device='cpu', seed=seed, force_reset=True,
        n_steps=640, learning_rate=rate,
    )
    # Loading the saved optimizer restores its old group hyperparameters.
    # Change only lr: keep all moments, step tensors, groups and counters.
    for group in model.policy.optimizer.param_groups:
        group['lr'] = rate
    return model
